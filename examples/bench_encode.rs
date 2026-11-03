//! Benchmark: hardware (VAAPI) vs software (libx264) encoding of identical
//! pre-blended RGB frames.
//!
//! Usage: cargo run --release --features video --example bench_encode -- <input.mp4> [frames]
//!
//! Decodes `frames` frames (default 300) from the input, blends them pairwise
//! with the GPU blender (so the encoder input matches the real pipeline), then
//! encodes the exact same frame data with each backend and reports wall time,
//! CPU time, ms/frame, and output size. Also writes the blended frames to
//! `ref.rgb` next to the outputs so PSNR can be computed afterwards with:
//!
//!   ffmpeg -i out.mp4 -f rawvideo -pix_fmt rgb24 -s 1920x1080 \
//!          -i ref.rgb -lavfi psnr -f null -

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anaglyph_rs::anaglyph::{anaglyph_type_to_matrix, AnaglyphType};
use anaglyph_rs::gpu::GpuBlender;
use anaglyph_rs::hwencode::{EncodeOptions, VaapiEncoder};
use ndarray::Array3;
use video_rs::decode::DecoderBuilder;
use video_rs::encode::{Encoder, Settings};
use video_rs::ffmpeg::Rational;
use video_rs::hwaccel::HardwareAccelerationDeviceType;
use video_rs::Time;

/// Process CPU time (user, sys) in seconds, from /proc/self/stat.
fn cpu_time() -> (f64, f64) {
    let mut stat = String::new();
    File::open("/proc/self/stat")
        .and_then(|mut f| f.read_to_string(&mut stat))
        .expect("read /proc/self/stat");
    // Skip the comm field (may contain spaces/parens) by starting after ')'.
    let fields: Vec<&str> = stat[stat.rfind(')').unwrap() + 1..]
        .split_whitespace()
        .collect();
    // fields[0] is state; utime is field 14 overall = index 11, stime 15 = 12.
    let utime: f64 = fields[11].parse().unwrap();
    let stime: f64 = fields[12].parse().unwrap();
    (utime / 100.0, stime / 100.0)
}

struct Timed {
    wall_ms_per_frame: f64,
    cpu_user_s: f64,
    cpu_sys_s: f64,
    bytes: u64,
    frames: usize,
}

fn report(name: &str, t: &Timed) {
    let cpu_total = t.cpu_user_s + t.cpu_sys_s;
    let wall_total = t.wall_ms_per_frame * t.frames as f64 / 1000.0;
    println!(
        "{name:<28} {:>7.2} ms/frame   cpu {:>5.2}s (u {:.2} + s {:.2})   \
         cpu/wall {:>4.1}x   {:.1} MiB",
        t.wall_ms_per_frame,
        cpu_total,
        t.cpu_user_s,
        t.cpu_sys_s,
        cpu_total / wall_total,
        t.bytes as f64 / (1024.0 * 1024.0),
    );
}

fn main() {
    video_rs::init().unwrap();
    let args: Vec<String> = std::env::args().collect();
    let input = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/tmp/opencode/anaglyph/testsrc1080.mp4");
    let wanted: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
    let out_dir = PathBuf::from(
        args.get(3)
            .map(String::as_str)
            .unwrap_or("/tmp/opencode/anaglyph"),
    );
    std::fs::create_dir_all(&out_dir).unwrap();

    // --- decode source frames ---
    let mut builder = DecoderBuilder::new(Path::new(input));
    if let Some(device) = [
        HardwareAccelerationDeviceType::VaApi,
        HardwareAccelerationDeviceType::Cuda,
    ]
    .iter()
    .find(|d| d.is_available())
    {
        builder = builder.with_hardware_acceleration(*device);
    }
    let mut decoder = builder.build().expect("open input");
    let (w, h) = decoder.size();
    let mut decoded: Vec<Vec<u8>> = Vec::new();
    while let Ok((_t, frame)) = decoder.decode() {
        decoded.push(frame.as_slice().unwrap().to_vec());
        if decoded.len() > wanted {
            break;
        }
    }
    println!(
        "decoded {} frames of {}x{}, blending {} pairs on GPU",
        decoded.len(),
        w,
        h,
        decoded.len() - 1
    );

    // --- blend pairwise exactly like the pipeline does ---
    let matrix = anaglyph_type_to_matrix(AnaglyphType::HalfColor);
    let mut blender = GpuBlender::new(w, h, matrix).expect("gpu blender");
    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(decoded.len() - 1);
    for pair in decoded.windows(2) {
        frames.push(blender.blend(&pair[1], &pair[0]).expect("blend"));
    }
    drop(decoded);
    let n = frames.len();
    let fps_tb = Rational(1, 30);

    // Reference dump of encoder input for PSNR measurement.
    let mut ref_file = File::create(out_dir.join("ref.rgb")).unwrap();
    for f in &frames {
        ref_file.write_all(f).unwrap();
    }
    drop(ref_file);

    // --- software encode (video-rs, x264 medium / CRF 23) ---
    let sw_path = out_dir.join("bench_only_sw.mp4");
    let _ = std::fs::remove_file(&sw_path);
    let (c0u, c0s) = cpu_time();
    let t = Instant::now();
    {
        let mut enc = Encoder::new(
            sw_path.as_path(),
            Settings::preset_h264_yuv420p(w as usize, h as usize, false),
        )
        .expect("software encoder");
        for (i, f) in frames.iter().enumerate() {
            let arr = Array3::from_shape_vec((h as usize, w as usize, 3), f.clone()).unwrap();
            enc.encode(&arr, Time::new(Some(i as i64), fps_tb))
                .expect("sw encode");
        }
        enc.finish().expect("sw finish");
    }
    let (c1u, c1s) = cpu_time();
    let wall = t.elapsed().as_secs_f64();
    report(
        "software x264 medium crf23",
        &Timed {
            wall_ms_per_frame: wall * 1000.0 / n as f64,
            cpu_user_s: c1u - c0u,
            cpu_sys_s: c1s - c0s,
            bytes: std::fs::metadata(&sw_path).unwrap().len(),
            frames: n,
        },
    );

    // --- hardware encode (h264_vaapi, CQP 23), several tunings ---
    let variants = [
        (
            "hw async_depth=2 ",
            EncodeOptions {
                qp: 23,
                gop_size: 12,
                async_depth: 2,
                low_power: false,
            },
        ),
        (
            "hw async_depth=4 ",
            EncodeOptions {
                qp: 23,
                gop_size: 12,
                async_depth: 4,
                low_power: false,
            },
        ),
        (
            "hw async_depth=8 ",
            EncodeOptions {
                qp: 23,
                gop_size: 12,
                async_depth: 8,
                low_power: false,
            },
        ),
        (
            "hw async_depth=16",
            EncodeOptions {
                qp: 23,
                gop_size: 12,
                async_depth: 16,
                low_power: false,
            },
        ),
        (
            "hw async=8 lowpwr",
            EncodeOptions {
                qp: 23,
                gop_size: 12,
                async_depth: 8,
                low_power: true,
            },
        ),
    ];
    for (label, opts) in variants {
        let hw_path = out_dir.join("bench_only_hw.mp4");
        let _ = std::fs::remove_file(&hw_path);
        // Time open + encode + flush together, like the software timing above.
        let (c0u, c0s) = cpu_time();
        let t = Instant::now();
        let result = (|| -> Result<(), String> {
            let mut enc = VaapiEncoder::new(&hw_path, w, h, fps_tb, opts)?;
            for (i, f) in frames.iter().enumerate() {
                enc.encode(f, i as i64)?;
            }
            enc.finish()
        })();
        let (c1u, c1s) = cpu_time();
        match result {
            Err(e) => println!("{label:<28} unavailable: {e}"),
            Ok(()) => {
                let wall = t.elapsed().as_secs_f64();
                report(
                    label,
                    &Timed {
                        wall_ms_per_frame: wall * 1000.0 / n as f64,
                        cpu_user_s: c1u - c0u,
                        cpu_sys_s: c1s - c0s,
                        bytes: std::fs::metadata(&hw_path).unwrap().len(),
                        frames: n,
                    },
                );
            }
        }
    }

    println!("frames: {n}, outputs + ref.rgb in {out_dir:?}");
}
