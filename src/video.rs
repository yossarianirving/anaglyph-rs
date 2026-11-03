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

use crate::anaglyph::{anaglyph_type_to_matrix, combine_slices, AnaglyphType, VideoDirection};
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

/// Depth of the queue between the decode thread and the blending loop. Also
/// bounded: decoding (especially on the GPU) outruns blending by far, and an
/// unbounded queue would pile up raw frames until memory runs out.
const DECODE_QUEUE_DEPTH: usize = 8;

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
    /// Encode one dense RGB24 frame, returning the buffer on success so the
    /// caller keeps ownership (the encode thread recycles it as the guard
    /// frame without copying).
    fn encode(
        &mut self,
        rgb: Vec<u8>,
        time: Time,
        shape: (usize, usize),
    ) -> Result<Vec<u8>, String> {
        match self {
            Backend::Software(encoder) => {
                let (height, width) = shape;
                let frame =
                    Array3::from_shape_vec((height, width, 3), rgb).map_err(|e| e.to_string())?;
                encoder.encode(&frame, time).map_err(|e| e.to_string())?;
                // Frames built by `from_shape_vec` always have a standard
                // layout, so the raw vec offset is None and the buffer is
                // returned as-is.
                Ok(frame.into_raw_vec_and_offset().0)
            }
            Backend::Hardware(encoder) => {
                let pts = time.into_value().unwrap_or(0);
                encoder.encode(&rgb, pts)?;
                Ok(rgb)
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

/// Summary of a finished conversion, for the caller to report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionReport {
    /// Number of blended frames encoded.
    pub frames: u64,
    /// Decoder used, e.g. `"cpu"` or `"VaApi hardware"`.
    pub decoder: String,
    /// Blender used, e.g. `"GPU (wgpu)"` or a CPU fallback with the reason.
    pub blender: String,
    /// Encoder used, e.g. `"VAAPI (h264_vaapi)"` or a fallback with the reason.
    pub encoder: String,
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
) -> Result<ConversionReport, String> {
    video_rs::init().map_err(|e| format!("failed to initialize ffmpeg: {e}"))?;

    let mut builder = DecoderBuilder::new(Path::new(video));
    let mut decoder_name = "cpu".to_string();
    // Note: GPU decoding downloads each frame back to system memory for the
    // blend step, which on many setups (including iGPUs) is slower than just
    // decoding on the CPU. Hardware decoding helps for high resolutions or
    // expensive codecs, so make it opt-in.
    if let DecoderChoice::Hardware = decoder_choice {
        if let Some(device) = HWACCEL_PREFERENCE
            .iter()
            .find(|device| device.is_available())
        {
            decoder_name = format!("{device:?} hardware");
            builder = builder.with_hardware_acceleration(*device);
        } else {
            decoder_name = "cpu (no compatible hardware device found)".into();
        }
    }
    let mut decoder = builder
        .build()
        .map_err(|e| format!("failed to open {video}: {e}"))?;

    let video_size = decoder.size();
    let (width, height) = (video_size.0 as usize, video_size.1 as usize);
    let decoder_time_base = decoder.time_base();

    let (backend, encoder_name) =
        create_backend(encoder_choice, video_out, width, height, decoder_time_base)?;

    let anaglyph_matrix = anaglyph_type_to_matrix(anaglyph_type);
    let (mut blender, mut blender_name) =
        match GpuBlender::new(video_size.0, video_size.1, anaglyph_matrix) {
            Ok(b) => (Some(b), "GPU (wgpu)".to_string()),
            Err(e) => (None, format!("CPU (rayon, GPU unavailable: {e})")),
        };

    let first_frame = decoder
        .decode()
        .map_err(|e| format!("failed to decode {video}: {e}"))?;

    // Decode on a background thread while this thread blends. The channel is
    // bounded (backpressure) and carries decode failures, so a broken stream
    // surfaces to the caller instead of silently truncating the output.
    let (tx, rx) = mpsc::sync_channel::<Result<(Time, Array3<u8>), String>>(DECODE_QUEUE_DEPTH);
    let decode_thread = thread::spawn(move || {
        // `decode_iter` yields forever: end of stream arrives as the
        // `DecodeExhausted` sentinel, which is a normal stop, not a failure.
        for frame in decoder.decode_iter() {
            match frame {
                Ok(time_frame) => {
                    if tx.send(Ok(time_frame)).is_err() {
                        break; // consumer is gone
                    }
                }
                Err(video_rs::error::Error::DecodeExhausted) => break,
                Err(e) => {
                    let _ = tx.send(Err(format!("decode error: {e}")));
                    break;
                }
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
    let mut decode_error: Option<String> = None;
    for item in rx {
        let (time, frame) = match item {
            Ok(time_frame) => time_frame,
            Err(e) => {
                decode_error = Some(e);
                break;
            }
        };
        let (previous_time, previous) = previous_frame.take().unwrap();
        let (left, right) = match direction {
            VideoDirection::Clockwise => (raw_rgb(&frame), raw_rgb(&previous)),
            VideoDirection::CounterClockwise => (raw_rgb(&previous), raw_rgb(&frame)),
        };

        let blended = match &mut blender {
            Some(gpu) => gpu.blend(left, right),
            None => Ok(blend_cpu(left, right, anaglyph_matrix)),
        };
        let rgb = match blended {
            Ok(rgb) => rgb,
            Err(e) => {
                // GPU device lost or readback failed mid-run: drop to CPU
                // blending for this and all remaining frames rather than
                // killing the conversion.
                blender = None;
                blender_name = format!("CPU (rayon, GPU blend failed mid-run: {e})");
                blend_cpu(left, right, anaglyph_matrix)
            }
        };

        let encode_time = timestamp_repair.repair(previous_time);
        if job_tx.send(Job::Frame(rgb, encode_time)).is_err() {
            // Encode thread died; its error is collected after join below.
            break;
        }
        frame_count += 1;
        previous_frame = Some((time, frame));
    }
    // The `for` loop consumed `rx`; dropping it here (implicit at break)
    // releases the decode thread if it was blocked on a full queue.

    // Give the muxer a packet beyond the final real frame: the MP4 muxer
    // cannot derive a duration for the very last packet and ends the track at
    // its PTS, which makes strict readers drop the final frame. One duplicate
    // frame at the end lets every real frame get a duration; the duplicate
    // itself lands at the track end and is invisible during playback.
    if let Some(guard_time) = timestamp_repair.next_time() {
        let _ = job_tx.send(Job::Guard(guard_time));
    }
    drop(job_tx);

    let encode_result = match encode_thread.join() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(format!("encoding failed: {e}")),
        Err(_) => Err("encode thread panicked".into()),
    };
    // The channel receiver is gone, so the decode thread is unwinding too.
    let _ = decode_thread.join();

    encode_result?;
    if let Some(e) = decode_error {
        return Err(format!(
            "decoding stopped early after {frame_count} frames: {e}"
        ));
    }
    if frame_count == 0 {
        return Err(format!(
            "{video} contains fewer than two frames; nothing to blend"
        ));
    }

    Ok(ConversionReport {
        frames: frame_count,
        decoder: decoder_name,
        blender: blender_name,
        encoder: encoder_name,
    })
}

/// Move a backend onto its own thread. The returned sender accepts jobs;
/// drop it to end the thread, whose `join()` yields the final result.
fn spawn_encoder(
    backend: Backend,
    shape: (usize, usize),
) -> (SyncSender<Job>, JoinHandle<Result<(), String>>) {
    let (job_tx, job_rx) = mpsc::sync_channel::<Job>(ENCODE_QUEUE_DEPTH);
    let handle = thread::spawn(move || {
        let mut backend = backend;
        let mut last_frame: Option<Vec<u8>> = None;
        for job in job_rx.iter() {
            match job {
                Job::Frame(rgb, time) => {
                    last_frame = Some(backend.encode(rgb, time, shape)?);
                }
                Job::Guard(time) => {
                    if let Some(rgb) = last_frame.take() {
                        backend.encode(rgb, time, shape)?;
                    }
                }
            }
        }
        backend.finish()
    });
    (job_tx, handle)
}

/// Create the encoding backend for the output file, along with a
/// human-readable description of the chosen backend.
fn create_backend(
    choice: EncoderChoice,
    video_out: &str,
    width: usize,
    height: usize,
    decoder_time_base: Rational,
) -> Result<(Backend, String), String> {
    let hardware = || -> Result<(Backend, String), String> {
        let encoder = VaapiEncoder::new(
            Path::new(video_out),
            width as u32,
            height as u32,
            decoder_time_base,
            EncodeOptions::default(),
        )?;
        Ok((
            Backend::Hardware(Box::new(encoder)),
            "VAAPI (h264_vaapi)".to_string(),
        ))
    };
    let software = || -> Result<(Backend, String), String> {
        let settings = Settings::preset_h264_yuv420p(width, height, false);
        let encoder = Encoder::new(Path::new(video_out), settings)
            .map_err(|e| format!("failed to create software encoder: {}", e))?;
        Ok((
            Backend::Software(Box::new(encoder)),
            "software (libx264)".to_string(),
        ))
    };

    match choice {
        EncoderChoice::Software => software(),
        EncoderChoice::Hardware => hardware(),
        EncoderChoice::Auto => match hardware() {
            Ok(ok) => Ok(ok),
            Err(e) => software().map(|(backend, _)| {
                (
                    backend,
                    format!("software (libx264, VAAPI unavailable: {e})"),
                )
            }),
        },
    }
}

/// CPU fallback for blending two dense RGB24 frames, parallelized with rayon.
fn blend_cpu(left: &[u8], right: &[u8], anaglyph_matrix: &[[f32; 9]; 2]) -> Vec<u8> {
    let mut out = vec![0u8; left.len()];
    out.par_chunks_mut(3)
        .zip(left.par_chunks_exact(3))
        .zip(right.par_chunks_exact(3))
        .for_each(|((out, left), right)| combine_slices(left, right, out, anaglyph_matrix));
    out
}

/// Get a contiguous RGB24 byte slice from a decoded frame.
///
/// Invariant, not input validation: video-rs hands out densely packed,
/// `Vec`-backed `Array3<u8>` frames, so `as_slice` cannot fail.
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

#[cfg(test)]
mod tests {
    use super::*;
    use video_rs::ffmpeg::ffi;

    fn tb() -> Rational {
        Rational(1, 30)
    }

    #[test]
    fn repair_passes_valid_timestamps_through() {
        let mut repair = TimestampRepair::new(tb());
        assert_eq!(
            repair.repair(Time::new(Some(0), tb())).into_value(),
            Some(0)
        );
        assert_eq!(
            repair.repair(Time::new(Some(2), tb())).into_value(),
            Some(2)
        );
    }

    #[test]
    fn repair_synthesizes_missing_timestamps_from_observed_step() {
        let mut repair = TimestampRepair::new(tb());
        repair.repair(Time::new(Some(10), tb()));
        repair.repair(Time::new(Some(12), tb())); // observed step: 2
        assert_eq!(repair.repair(Time::new(None, tb())).into_value(), Some(14));
        assert_eq!(
            repair
                .repair(Time::new(Some(ffi::AV_NOPTS_VALUE), tb()))
                .into_value(),
            Some(16)
        );
        assert_eq!(repair.next_time().and_then(Time::into_value), Some(18));
    }

    #[test]
    fn repair_without_known_step_leaves_timestamp_untouched() {
        let mut repair = TimestampRepair::new(tb());
        repair.repair(Time::new(Some(5), tb()));
        // Only one observation so far: no step to extrapolate from.
        assert!(!repair.repair(Time::new(None, tb())).has_value());
        assert!(repair.next_time().is_none());
    }
}
