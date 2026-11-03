//! Stage timing for gif conversion: decode, blend (CPU vs GPU), encode.
use std::io::Cursor;
use std::time::Instant;

use anaglyph_rs::anaglyph::{anaglyph_type_to_matrix, AnaglyphType};
use anaglyph_rs::gif;
use anaglyph_rs::gpu::GpuBlender;
use image::{AnimationDecoder, Frame, ImageDecoder};

fn main() {
    let path = std::env::args().nth(1).unwrap_or("test/3drot.gif".into());
    let data = std::fs::read(&path).unwrap();

    // decode
    let t = Instant::now();
    let decoder = image::codecs::gif::GifDecoder::new(Cursor::new(&data)).unwrap();
    let (w, h) = decoder.dimensions();
    let mut frames: Vec<Vec<u8>> = Vec::new();
    let mut delay = image::Delay::from_numer_denom_ms(0, 1);
    for (i, f) in decoder.into_frames().enumerate() {
        let f = f.unwrap();
        if i == 0 {
            delay = f.delay();
        }
        frames.push(f.into_buffer().into_raw());
    }
    let decode_t = t.elapsed().as_secs_f64();
    let n = frames.len();
    println!(
        "decode: {n} frames {w}x{h} in {decode_t:.2}s = {:.2} ms/frame",
        decode_t * 1000.0 / n as f64
    );

    // blend on GPU
    let matrix = anaglyph_type_to_matrix(AnaglyphType::Color);
    let mut gpu = GpuBlender::new(w, h, matrix).unwrap();
    let t = Instant::now();
    let mut blended: Vec<Vec<u8>> = Vec::with_capacity(n - 1);
    for pair in frames.windows(2) {
        blended.push(gpu.blend_rgba(&pair[1], &pair[0]).expect("blend"));
    }
    let blend_t = t.elapsed().as_secs_f64();
    println!(
        "blend GPU:  in {blend_t:.2}s = {:.2} ms/frame",
        blend_t * 1000.0 / n as f64
    );

    // encode the real blended frames
    let new_frames: Vec<Frame> = blended
        .into_iter()
        .map(|px| Frame::from_parts(image::RgbaImage::from_vec(w, h, px).unwrap(), 0, 0, delay))
        .collect();
    let t = Instant::now();
    let mut buf = Vec::new();
    {
        let mut enc = image::codecs::gif::GifEncoder::new_with_speed(&mut buf, 10);
        enc.set_repeat(image::codecs::gif::Repeat::Infinite)
            .unwrap();
        enc.encode_frames(new_frames).unwrap();
    }
    let enc_t = t.elapsed().as_secs_f64();
    println!(
        "encode: out in {enc_t:.2}s = {:.2} ms/frame",
        enc_t * 1000.0 / n as f64
    );

    // full pipeline for reference
    let t = Instant::now();
    let full = gif::convert_gif_buffer_to_anaglyph(
        &data,
        anaglyph_rs::anaglyph::VideoDirection::Clockwise,
        AnaglyphType::Color,
    )
    .expect("convert");
    println!(
        "full convert_gif_buffer_to_anaglyph: {:.2}s ({} MiB out)",
        t.elapsed().as_secs_f64(),
        full.len() / 1048576
    );
}
