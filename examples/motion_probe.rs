//! Diagnostic probe for the camera-direction analyser: prints the raw
//! full-frame and centre-window drifts and the selected regime for every
//! frame pair, so a misdetected direction can be traced back to the data.
//!
//! ```text
//! cargo run --release --features video --example motion_probe -- out.mp4 motion.tsv
//! ```

use std::error::Error;
use std::io::Write;
use std::path::Path;

use video_rs::decode::DecoderBuilder;

use anaglyph_rs::motion::{MotionEstimator, MotionSample};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let Some(input) = args.next() else {
        return Err("usage: motion_probe <video> [out.tsv]".into());
    };
    video_rs::init()?;
    let mut decoder = DecoderBuilder::new(Path::new(&input)).build()?;
    let (width, height) = decoder.size();

    let mut out: Box<dyn Write> = match args.next() {
        Some(path) => Box::new(std::io::BufWriter::new(std::fs::File::create(path)?)),
        None => Box::new(std::io::stdout()),
    };
    writeln!(
        out,
        "frame\ttime\tfull_dx\tfull_ok\tcenter_dx\tcenter_ok\tregime"
    )?;

    let mut est = MotionEstimator::new(width as usize, height as usize);
    let mut frame_index: u64 = 0;
    let mut emit = |index: u64, time: f64, sample: &MotionSample| -> Result<(), Box<dyn Error>> {
        let ok = |v: f32| v.is_finite();
        writeln!(
            out,
            "{index}\t{time:.3}\t{}\t{}\t{}\t{}\t{}",
            num(sample.full_dx),
            ok(sample.full_dx),
            num(sample.center_dx),
            ok(sample.center_dx),
            sample.regime.name(),
        )?;
        Ok(())
    };

    let (time, frame) = decoder.decode()?;
    emit(
        frame_index,
        seconds(time),
        &est.observe(frame.as_slice().unwrap()),
    )?;
    frame_index += 1;
    for frame in decoder.decode_iter() {
        // End of stream arrives as an exhausted sentinel, not a failure.
        let (time, frame) = match frame {
            Ok(pair) => pair,
            Err(video_rs::error::Error::DecodeExhausted)
            | Err(video_rs::error::Error::ReadExhausted) => break,
            Err(e) => return Err(e.into()),
        };
        emit(
            frame_index,
            seconds(time),
            &est.observe(frame.as_slice().unwrap()),
        )?;
        frame_index += 1;
    }
    eprintln!("probed {frame_index} frames of {width}x{height}");
    Ok(())
}

/// Decoder timestamp in seconds.
fn seconds(time: video_rs::Time) -> f64 {
    time.as_secs_f64()
}

/// Formats a drift for TSV; unmatched measurements print as `nan`.
fn num(v: f32) -> String {
    if v.is_finite() {
        format!("{v:.3}")
    } else {
        "nan".to_string()
    }
}
