//! Stage timing: VAAPI decode rate and GPU blend rate.
use std::time::Instant;
use anaglyph_rs::anaglyph::{anaglyph_type_to_matrix, AnaglyphType};
use anaglyph_rs::gpu::GpuBlender;
use video_rs::decode::DecoderBuilder;
use video_rs::hwaccel::HardwareAccelerationDeviceType;

fn main() {
    video_rs::init().unwrap();
    let input = std::env::args().nth(1).unwrap_or("/tmp/opencode/anaglyph/testsrc1080.mp4".into());

    // decode only (to RGB24 ndarray)
    let mut b = DecoderBuilder::new(std::path::Path::new(&input));
    let hw = std::env::var("HWDECODE").unwrap_or_default() == "1";
    if hw {
        if let Some(d) = [HardwareAccelerationDeviceType::VaApi].iter().find(|d| d.is_available()) {
            b = b.with_hardware_acceleration(*d);
        }
    }
    let mut dec = b.build().unwrap();
    let (w, h) = dec.size();
    let t = Instant::now();
    let mut n = 0usize;
    let mut sample: Vec<Vec<u8>> = Vec::new();
    let mut keep = false;
    while let Ok((_t2, frame)) = dec.decode() {
        n += 1;
        if n > 50 && n <= 150 { keep = true; }
        if keep && sample.len() < 100 { sample.push(frame.as_slice().unwrap().to_vec()); }
    }
    let dt = t.elapsed().as_secs_f64();
    println!("decode: {n} frames {w}x{h} in {dt:.2}s = {:.2} ms/frame", dt * 1000.0 / n as f64);

    // blend only (warm GPU), on the decoded samples
    let m = anaglyph_type_to_matrix(AnaglyphType::HalfColor);
    let mut gpu = GpuBlender::new(w, h, m).unwrap();
    let t = Instant::now();
    let mut out = Vec::new();
    for pair in sample.windows(2) {
        out = gpu.blend(&pair[1], &pair[0]);
    }
    let dt = t.elapsed().as_secs_f64();
    println!("blend: 100 frames in {dt:.2}s = {:.2} ms/frame (out {} B)", dt * 10.0, out.len());
}
