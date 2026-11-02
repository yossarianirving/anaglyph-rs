use ndarray::Array3;
use rayon::prelude::*;
use std::path::Path;
use std::sync::mpsc;
use std::thread;

use video_rs::decode::DecoderBuilder;
use video_rs::encode::{Encoder, Settings};
use video_rs::ffmpeg::Rational;
use video_rs::hwaccel::HardwareAccelerationDeviceType;
use video_rs::Time;

use crate::anaglyph::{
    anaglyph_type_to_matrix, combine_slices, AnaglyphType, VideoDirection,
};
use crate::gpu::GpuBlender;

/// Device types to try for hardware-accelerated decoding, in order.
const HWACCEL_PREFERENCE: [HardwareAccelerationDeviceType; 3] = [
    HardwareAccelerationDeviceType::VaApi,
    HardwareAccelerationDeviceType::Cuda,
    HardwareAccelerationDeviceType::Vdpau,
];

pub fn convert_video_to_anaglyph(
    video: &str,
    video_out: &str,
    direction: VideoDirection,
    anaglyph_type: AnaglyphType,
) {
    video_rs::init().unwrap();

    let mut builder = DecoderBuilder::new(Path::new(video));
    if let Some(device) = HWACCEL_PREFERENCE
        .iter()
        .find(|device| device.is_available())
    {
        println!("Using {:?} hardware decoding", device);
        builder = builder.with_hardware_acceleration(*device);
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
    let encoder_settings = Settings::preset_h264_yuv420p(width, height, false);
    let mut encoder = match Encoder::new(Path::new(video_out), encoder_settings) {
        Ok(encoder) => encoder,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };

    println!(" {:?}", video_size);
    let decoder_time_base = decoder.time_base();
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

    // Decode on a background thread while this thread blends and encodes.
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

    let mut timestamp_repair = TimestampRepair::new(decoder_time_base);
    let mut previous_frame = Some(first_frame);
    let mut last_output: Option<ndarray::Array3<u8>> = None;
    let mut frame_count: u64 = 0;
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
        let new_frame = match Array3::from_shape_vec((height, width, 3), rgb) {
            Ok(frame) => frame,
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        };

        let encode_time = timestamp_repair.repair(previous_time);
        if let Err(e) = encoder.encode(&new_frame, encode_time) {
            eprintln!("Encode error: {}", e);
        }
        frame_count += 1;
        last_output = Some(new_frame);
        previous_frame = Some((time, frame));
    }

    // Give the muxer a packet beyond the final real frame: the MP4 muxer
    // cannot derive a duration for the very last packet and ends the track at
    // its PTS, which makes strict readers drop the final frame. One duplicate
    // frame at the end lets every real frame get a duration; the duplicate
    // itself lands at the track end and is invisible during playback.
    if let (Some(frame), Some(next_time)) = (last_output.as_ref(), timestamp_repair.next_time()) {
        if let Err(e) = encoder.encode(frame, next_time) {
            eprintln!("Encode error: {}", e);
        }
    }

    println!("Encoded {} frames", frame_count);
    encoder.finish().expect("failed to finish encoder");
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
