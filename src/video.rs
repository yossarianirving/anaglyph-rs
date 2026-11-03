use ndarray::Array3;
use rayon::prelude::*;
use std::path::Path;
use std::sync::mpsc::{self, SyncSender};
use std::thread::{self, JoinHandle};

use video_rs::decode::DecoderBuilder;
use video_rs::encode::{Encoder, Settings};
use video_rs::ffmpeg::Rational;
use video_rs::hwaccel::HardwareAccelerationDeviceType;
use video_rs::Time;

use crate::anaglyph::{
    anaglyph_type_to_matrix, combine_slices, AnaglyphType, VideoDirection,
};
use crate::gpu::GpuBlender;
use crate::hwencode::{EncodeOptions, VaapiEncoder};

/// Device types to try for hardware-accelerated decoding, in order.
const HWACCEL_PREFERENCE: [HardwareAccelerationDeviceType; 3] = [
    HardwareAccelerationDeviceType::VaApi,
    HardwareAccelerationDeviceType::Cuda,
    HardwareAccelerationDeviceType::Vdpau,
];

/// Depth of the queue between the blending thread and the encoding thread.
/// Caps memory in flight at roughly `queue * frame size` and provides
/// backpressure when the encoder falls behind.
const ENCODE_QUEUE_DEPTH: usize = 8;

/// Which encoder backend to use for the output video.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EncoderChoice {
    /// Prefer the hardware (VAAPI) encoder, fall back to software.
    Auto,
    /// Require the hardware (VAAPI) encoder; fail if unavailable.
    Hardware,
    /// Always use the software (libx264) encoder.
    Software,
}

/// The active encoding backend.
enum Backend {
    Software(Box<Encoder>),
    Hardware(Box<VaapiEncoder>),
}

impl Backend {
    /// Encode one dense RGB24 frame.
    fn encode(&mut self, rgb: &[u8], time: Time, shape: (usize, usize)) -> Result<(), String> {
        match self {
            Backend::Software(encoder) => {
                let (height, width) = shape;
                let frame = Array3::from_shape_vec((height, width, 3), rgb.to_vec())
                    .map_err(|e| e.to_string())?;
                encoder.encode(&frame, time).map_err(|e| e.to_string())
            }
            Backend::Hardware(encoder) => {
                let pts = time.into_value().unwrap_or(0);
                encoder.encode(rgb, pts)
            }
        }
    }

    fn finish(&mut self) -> Result<(), String> {
        match self {
            Backend::Software(encoder) => encoder.finish().map_err(|e| e.to_string()),
            Backend::Hardware(encoder) => encoder.finish(),
        }
    }
}

/// A unit of encoding work handed to the encode thread.
enum Job {
    /// Encode this RGB24 frame at the given timestamp.
    Frame(Vec<u8>, Time),
    /// Encode the previous frame again at the given (later) timestamp so the
    /// muxer can assign a duration to the final real frame (see call site).
    Guard(Time),
}

/// Which decoder mode to use for the input video.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecoderChoice {
    /// Decode on the CPU with FFmpeg's software decoder.
    Cpu,
    /// Decode with the GPU (VAAPI/CUDA/VDPAU) when available.
    Hardware,
}

pub fn convert_video_to_anaglyph(
    video: &str,
    video_out: &str,
    direction: VideoDirection,
    anaglyph_type: AnaglyphType,
    decoder_choice: DecoderChoice,
    encoder_choice: EncoderChoice,
) {
    video_rs::init().unwrap();

    let mut builder = DecoderBuilder::new(Path::new(video));
    // Note: GPU decoding downloads each frame back to system memory for the
    // blend step, which on many setups (including iGPUs) is slower than just
    // decoding on the CPU. Hardware decoding helps for high resolutions or
    // expensive codecs, so make it opt-in.
    if let DecoderChoice::Hardware = decoder_choice {
        if let Some(device) = HWACCEL_PREFERENCE
            .iter()
            .find(|device| device.is_available())
        {
            println!("Using {:?} hardware decoding", device);
            builder = builder.with_hardware_acceleration(*device);
        }
    }
    let mut decoder = match builder.build() {
        Ok(decoder) => decoder,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };

    let video_size = decoder.size();
    let (width, height) = (video_size.0 as usize, video_size.1 as usize);
    let decoder_time_base = decoder.time_base();

    let backend = match create_backend(encoder_choice, video_out, width, height, decoder_time_base)
    {
        Ok(backend) => backend,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };

    println!(" {:?}", video_size);
    let anaglyph_matrix = anaglyph_type_to_matrix(anaglyph_type);
    let mut blender = match GpuBlender::new(video_size.0, video_size.1, anaglyph_matrix) {
        Ok(blender) => Some(blender),
        Err(e) => {
            eprintln!("GPU blending unavailable ({}), falling back to CPU", e);
            None
        }
    };

    let first_frame = match decoder.decode() {
        Ok(frame) => frame,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };

    // Decode on a background thread while this thread blends.
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for frame in decoder.decode_iter() {
            match frame {
                Ok(time_frame) => {
                    if tx.send(time_frame).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    // Encode on a second background thread so CPU conversion and the encoder
    // overlap with GPU blending of the next frame. The bounded queue keeps
    // memory in flight in check.
    let (job_tx, encode_thread) = spawn_encoder(backend, (height, width));

    let mut timestamp_repair = TimestampRepair::new(decoder_time_base);
    let mut previous_frame = Some(first_frame);
    let mut frame_count: u64 = 0;
    let mut failure: Option<String> = None;
    for (time, frame) in rx {
        let (previous_time, previous) = previous_frame.take().unwrap();
        let (left, right) = match direction {
            VideoDirection::Clockwise => (raw_rgb(&frame), raw_rgb(&previous)),
            VideoDirection::CounterClockwise => (raw_rgb(&previous), raw_rgb(&frame)),
        };

        let rgb = match &mut blender {
            Some(gpu) => gpu.blend(left, right),
            None => blend_cpu(left, right, anaglyph_matrix),
        };

        let encode_time = timestamp_repair.repair(previous_time);
        if job_tx.send(Job::Frame(rgb, encode_time)).is_err() {
            // Encode thread died; its error is collected after join below.
            break;
        }
        frame_count += 1;
        previous_frame = Some((time, frame));
    }

    // Give the muxer a packet beyond the final real frame: the MP4 muxer
    // cannot derive a duration for the very last packet and ends the track at
    // its PTS, which makes strict readers drop the final frame. One duplicate
    // frame at the end lets every real frame get a duration; the duplicate
    // itself lands at the track end and is invisible during playback.
    if failure.is_none() {
        if let Some(guard_time) = timestamp_repair.next_time() {
            let _ = job_tx.send(Job::Guard(guard_time));
        }
    }
    drop(job_tx);

    match encode_thread.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => failure = Some(e),
        Err(_) => failure = Some("encode thread panicked".into()),
    }
    match failure {
        None => println!("Encoded {} frames", frame_count),
        Some(e) => eprintln!("Encode error: {}", e),
    }
}

/// Move a backend onto its own thread. The returned sender accepts jobs;
/// drop it to end the thread, whose `join()` yields the final result.
fn spawn_encoder(backend: Backend, shape: (usize, usize)) -> (SyncSender<Job>, JoinHandle<Result<(), String>>) {
    let (job_tx, job_rx) = mpsc::sync_channel::<Job>(ENCODE_QUEUE_DEPTH);
    let handle = thread::spawn(move || {
        let mut backend = backend;
        let mut last_frame: Option<Vec<u8>> = None;
        for job in job_rx.iter() {
            match job {
                Job::Frame(rgb, time) => {
                    backend.encode(&rgb, time, shape)?;
                    last_frame = Some(rgb);
                }
                Job::Guard(time) => {
                    if let Some(rgb) = last_frame.as_ref() {
                        backend.encode(rgb, time, shape)?;
                    }
                }
            }
        }
        backend.finish()
    });
    (job_tx, handle)
}

/// Create the encoding backend for the output file.
fn create_backend(
    choice: EncoderChoice,
    video_out: &str,
    width: usize,
    height: usize,
    decoder_time_base: Rational,
) -> Result<Backend, String> {
    let hardware = || -> Result<Backend, String> {
        let encoder = VaapiEncoder::new(
            Path::new(video_out),
            width as u32,
            height as u32,
            decoder_time_base,
            EncodeOptions::default(),
        )?;
        println!("Using VAAPI hardware encoding");
        Ok(Backend::Hardware(Box::new(encoder)))
    };
    let software = || -> Result<Backend, String> {
        let settings = Settings::preset_h264_yuv420p(width, height, false);
        let encoder = Encoder::new(Path::new(video_out), settings)
            .map_err(|e| format!("failed to create software encoder: {}", e))?;
        println!("Using software (libx264) encoding");
        Ok(Backend::Software(Box::new(encoder)))
    };

    match choice {
        EncoderChoice::Software => software(),
        EncoderChoice::Hardware => hardware(),
        EncoderChoice::Auto => match hardware() {
            Ok(backend) => Ok(backend),
            Err(e) => {
                eprintln!("Hardware encoding unavailable ({}), falling back to software", e);
                software()
            }
        },
    }
}

/// CPU fallback for blending two dense RGB24 frames, parallelized with rayon.
fn blend_cpu(left: &[u8], right: &[u8], anaglyph_matrix: &[[f32; 9]; 2]) -> Vec<u8> {
    let mut out = vec![0u8; left.len()];
    out.par_chunks_mut(3)
        .zip(left.par_chunks_exact(3))
        .zip(right.par_chunks_exact(3))
        .for_each(|((out, left), right)| {
            combine_slices(left, right, out, anaglyph_matrix)
        });
    out
}

/// Get a contiguous RGB24 byte slice from a decoded frame.
fn raw_rgb(frame: &ndarray::Array3<u8>) -> &[u8] {
    frame.as_slice().expect("decoded frames are contiguous")
}

/// Repairs missing frame timestamps.
///
/// The last packets of a stream (common with B-frames) can have no DTS, and
/// the muxer silently drops frames encoded with such a timestamp. Instead of
/// dropping the frame, synthesize its timestamp from the last valid one and
/// the observed timestamp step.
struct TimestampRepair {
    time_base: Rational,
    last_value: Option<i64>,
    step: Option<i64>,
}

impl TimestampRepair {
    fn new(time_base: Rational) -> Self {
        TimestampRepair {
            time_base,
            last_value: None,
            step: None,
        }
    }

    /// The timestamp one frame step past the last repaired one, if both the
    /// last value and an observed step are known.
    fn next_time(&self) -> Option<Time> {
        match (self.last_value, self.step) {
            (Some(last), Some(step)) => Some(Time::new(Some(last + step), self.time_base)),
            _ => None,
        }
    }

    fn repair(&mut self, time: Time) -> Time {
        if time.has_no_pts() || !time.has_value() {
            match (self.last_value, self.step) {
                (Some(last), Some(step)) => {
                    let repaired = last + step;
                    self.last_value = Some(repaired);
                    Time::new(Some(repaired), self.time_base)
                }
                _ => time,
            }
        } else {
            if let Some(value) = time.into_value() {
                if let Some(last) = self.last_value {
                    if value > last {
                        self.step = Some(value - last);
                    }
                }
                self.last_value = Some(value);
            }
            time
        }
    }
}
