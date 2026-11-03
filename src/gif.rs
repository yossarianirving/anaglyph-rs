use crate::anaglyph::{anaglyph_type_to_matrix, combine_slices, AnaglyphType, VideoDirection};
use image::codecs::gif::{GifDecoder, GifEncoder};
use image::{AnimationDecoder, Frame, ImageDecoder, RgbaImage};
use std::fs::File;
use std::io::{Cursor, Read, Write};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "gpu")]
use crate::gpu::GpuBlender;

/// Core buffer-based function for GIF anaglyph conversion.
/// This can be called from WASM or anywhere you have GIF data in memory.
///
/// Frames are blended pairwise (frame N with frame N-1), so the output has
/// one frame fewer than the input; each output frame carries the delay of
/// its newer source frame, preserving the original animation timing.
///
/// Native builds with the `gpu` feature blend on the GPU (falling back to
/// the CPU when no device is available); other targets blend on the CPU.
/// On wasm, prefer [`convert_gif_buffer_to_anaglyph_async`], which can use
/// WebGPU without blocking the browser's event loop.
pub fn convert_gif_buffer_to_anaglyph(
    gif_data: &[u8],
    direction: VideoDirection,
    anaglyph_type: AnaglyphType,
) -> Result<Vec<u8>, String> {
    let cursor_in = Cursor::new(gif_data);
    let decoder = GifDecoder::new(cursor_in).map_err(|e| format!("failed to decode gif: {e}"))?;
    let gif_size = decoder.dimensions();
    let mut frames = decoder.into_frames();

    let anaglyph_matrix = anaglyph_type_to_matrix(anaglyph_type);

    let first_frame = frames
        .next()
        .ok_or("gif contains no frames")?
        .map_err(|e| format!("failed to decode frame: {e}"))?
        .into_buffer(); // RgbaImage
    let mut previous_frame = first_frame;

    // GPU blending when available (native builds with the `gpu` feature);
    // any setup failure (no adapter, oversized frame) falls back to CPU.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    let mut gpu = GpuBlender::new(gif_size.0, gif_size.1, anaglyph_matrix).ok();

    let mut new_frames = Vec::<Frame>::new();
    for frame in frames {
        let frame = frame.map_err(|e| format!("failed to decode frame: {e}"))?;
        // Take the delay before consuming the frame into a buffer: every
        // source frame can have its own timing.
        let delay = frame.delay();
        let rgba_frame = frame.into_buffer(); // RgbaImage

        let (left, right) = match direction {
            VideoDirection::Clockwise => (
                rgba_frame.as_raw().as_slice(),
                previous_frame.as_raw().as_slice(),
            ),
            VideoDirection::CounterClockwise => (
                previous_frame.as_raw().as_slice(),
                rgba_frame.as_raw().as_slice(),
            ),
        };

        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        let blended = match &mut gpu {
            Some(blender) => blender.blend_rgba(left, right)?,
            None => combine_rgba(left, right, anaglyph_matrix),
        };
        #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
        let blended = combine_rgba(left, right, anaglyph_matrix);

        let new_rgba_frame = RgbaImage::from_vec(gif_size.0, gif_size.1, blended)
            .ok_or("blended frame has unexpected size")?;
        new_frames.push(Frame::from_parts(new_rgba_frame, 0, 0, delay));
        previous_frame = rgba_frame;
    }

    if new_frames.is_empty() {
        return Err(
            "gif contains only one frame; anaglyph conversion needs at least two frames to pair"
                .into(),
        );
    }

    encode_anaglyph(new_frames)
}

/// Async variant of [`convert_gif_buffer_to_anaglyph`] for wasm: blends
/// frames on WebGPU without blocking the event loop when the `gpu` feature
/// is enabled, and stays on the CPU otherwise. Errors are returned as
/// strings since a wasm panic would silently kill a worker.
#[cfg(target_arch = "wasm32")]
pub async fn convert_gif_buffer_to_anaglyph_async(
    gif_data: &[u8],
    direction: VideoDirection,
    anaglyph_type: AnaglyphType,
) -> Result<Vec<u8>, String> {
    let cursor_in = Cursor::new(gif_data);
    let decoder = GifDecoder::new(cursor_in).map_err(|e| format!("failed to decode gif: {e}"))?;
    let gif_size = decoder.dimensions();
    let mut frames = decoder.into_frames();

    let anaglyph_matrix = anaglyph_type_to_matrix(anaglyph_type);

    let first_frame = frames
        .next()
        .ok_or("gif contains no frames")?
        .map_err(|e| format!("failed to decode frame: {e}"))?
        .into_buffer();
    let mut previous_frame = first_frame;

    // Fall back to CPU blending when WebGPU is unavailable; report the
    // reason on the JS console so the fallback is visible during debugging.
    #[cfg(feature = "gpu")]
    let mut gpu = match GpuBlender::new_async(gif_size.0, gif_size.1, anaglyph_matrix).await {
        Ok(blender) => {
            #[cfg(feature = "wasm")]
            web_sys::console::log_1(&"anaglyph: blending frames on WebGPU".into());
            Some(blender)
        }
        Err(e) => {
            #[cfg(feature = "wasm")]
            web_sys::console::warn_1(
                &format!("anaglyph: WebGPU unavailable ({e}), using CPU blending").into(),
            );
            None
        }
    };

    let mut new_frames = Vec::<Frame>::new();
    for frame in frames {
        let frame = frame.map_err(|e| format!("failed to decode frame: {e}"))?;
        let delay = frame.delay();
        let rgba_frame = frame.into_buffer();

        let (left, right) = match direction {
            VideoDirection::Clockwise => (
                rgba_frame.as_raw().as_slice(),
                previous_frame.as_raw().as_slice(),
            ),
            VideoDirection::CounterClockwise => (
                previous_frame.as_raw().as_slice(),
                rgba_frame.as_raw().as_slice(),
            ),
        };

        #[cfg(feature = "gpu")]
        let blended = match &mut gpu {
            Some(blender) => blender.blend_rgba_async(left, right).await?,
            None => combine_rgba(left, right, anaglyph_matrix),
        };
        #[cfg(not(feature = "gpu"))]
        let blended = combine_rgba(left, right, anaglyph_matrix);

        let new_rgba_frame = RgbaImage::from_vec(gif_size.0, gif_size.1, blended)
            .ok_or("blended frame has unexpected size")?;
        new_frames.push(Frame::from_parts(new_rgba_frame, 0, 0, delay));
        previous_frame = rgba_frame;
    }

    if new_frames.is_empty() {
        return Err(
            "gif contains only one frame; anaglyph conversion needs at least two frames to pair"
                .into(),
        );
    }

    encode_anaglyph(new_frames)
}

/// CPU blend of two dense RGBA frames with the anaglyph matrix. Pixels that
/// are fully transparent in both inputs stay transparent; all other pixels
/// become opaque. Parallelized with rayon when the `parallel` feature is on.
fn combine_rgba(left: &[u8], right: &[u8], anaglyph_matrix: &[[f32; 9]; 2]) -> Vec<u8> {
    debug_assert_eq!(left.len(), right.len());
    let mut out = vec![0u8; left.len()];

    #[cfg(feature = "parallel")]
    out.par_chunks_mut(4)
        .zip(left.par_chunks_exact(4))
        .zip(right.par_chunks_exact(4))
        .for_each(|((out, left), right)| combine_rgba_pixel(left, right, out, anaglyph_matrix));
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(4)
        .zip(left.chunks_exact(4))
        .zip(right.chunks_exact(4))
        .for_each(|((out, left), right)| combine_rgba_pixel(left, right, out, anaglyph_matrix));

    out
}

/// Blend one RGBA pixel pair (`out` is four bytes, zero-initialized).
///
/// Simplification: RGB is blended regardless of the sources' alpha values
/// (no alpha-weighted compositing), which matches the GPU reference; only
/// the both-fully-transparent case stays transparent, everything else comes
/// out opaque.
fn combine_rgba_pixel(left: &[u8], right: &[u8], out: &mut [u8], anaglyph_matrix: &[[f32; 9]; 2]) {
    if left[3] == 0 && right[3] == 0 {
        return; // left as [0, 0, 0, 0]: fully transparent
    }
    combine_slices(&left[..3], &right[..3], &mut out[..3], anaglyph_matrix);
    out[3] = 255;
}

/// Mux blended frames into an animated gif, looping forever.
fn encode_anaglyph(new_frames: Vec<Frame>) -> Result<Vec<u8>, String> {
    let mut out_buffer = Vec::new();
    {
        // The `speed` argument selects the palette-optimization effort, not
        // a fixed frame rate: `encode_frames` honors each frame's own delay.
        let mut encoder = GifEncoder::new_with_speed(&mut out_buffer, 10);
        encoder
            .set_repeat(image::codecs::gif::Repeat::Infinite)
            .map_err(|e| format!("failed to set gif repeat: {e}"))?;
        encoder
            .encode_frames(new_frames)
            .map_err(|e| format!("failed to encode gif frames: {e}"))?;
    }
    Ok(out_buffer)
}

/// Wrapper function for file-based usage.
/// Retains the original signature and calls the buffer-based function.
pub fn convert_gif_to_anaglyph(
    gif: &str,
    gif_out: &str,
    direction: VideoDirection,
    anaglyph_type: AnaglyphType,
) -> Result<(), String> {
    // Read input file into buffer
    let mut input_file = File::open(gif).map_err(|e| format!("failed to open {gif}: {e}"))?;
    let mut gif_data = Vec::new();
    input_file
        .read_to_end(&mut gif_data)
        .map_err(|e| format!("failed to read {gif}: {e}"))?;

    // Call buffer-based function
    let out_data = convert_gif_buffer_to_anaglyph(&gif_data, direction, anaglyph_type)?;

    // Write output buffer to file
    let mut output_file =
        File::create(gif_out).map_err(|e| format!("failed to create {gif_out}: {e}"))?;
    output_file
        .write_all(&out_data)
        .map_err(|e| format!("failed to write {gif_out}: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_rgba_both_transparent_stays_transparent() {
        let matrix = anaglyph_type_to_matrix(AnaglyphType::Color);
        let left = [10, 20, 30, 0];
        let right = [40, 50, 60, 0];
        let out = combine_rgba(&left, &right, matrix);
        assert_eq!(out, vec![0, 0, 0, 0]);
    }

    #[test]
    fn combine_rgba_partially_transparent_becomes_opaque() {
        let matrix = anaglyph_type_to_matrix(AnaglyphType::Color);
        let left = [10, 20, 30, 0];
        let right = [40, 50, 60, 255];
        let out = combine_rgba(&left, &right, matrix);
        // Color matrix: red from left, green/blue from right, alpha forced.
        assert_eq!(out, vec![10, 50, 60, 255]);
    }

    #[test]
    fn decoding_garbage_is_an_error_not_a_panic() {
        let err = convert_gif_buffer_to_anaglyph(
            b"not a gif",
            VideoDirection::Clockwise,
            AnaglyphType::Color,
        );
        assert!(err.is_err());
    }

    fn solid(color: image::Rgba<u8>, delay_ms: u32) -> Frame {
        let img = RgbaImage::from_pixel(4, 4, color);
        Frame::from_parts(img, 0, 0, image::Delay::from_numer_denom_ms(delay_ms, 1))
    }

    #[test]
    fn variable_frame_delays_are_preserved() {
        // Build a 3-frame gif with delays 100/200/300ms. Pairing produces
        // two output frames that must carry the *newer* frame's delay, not
        // the first frame's delay (the old bug).
        let mut buf = Vec::new();
        {
            let mut enc = GifEncoder::new(&mut buf);
            enc.encode_frames(vec![
                solid(image::Rgba([255, 0, 0, 255]), 100),
                solid(image::Rgba([0, 255, 0, 255]), 200),
                solid(image::Rgba([0, 0, 255, 255]), 300),
            ])
            .unwrap();
        }

        let out =
            convert_gif_buffer_to_anaglyph(&buf, VideoDirection::Clockwise, AnaglyphType::Color)
                .expect("conversion");

        let dec = GifDecoder::new(Cursor::new(&out)).unwrap();
        let delays: Vec<u32> = dec
            .into_frames()
            .map(|f| f.unwrap().delay().numer_denom_ms().0)
            .collect();
        assert_eq!(delays, vec![200, 300]);
    }

    #[test]
    fn single_frame_gif_is_a_clean_error() {
        let mut buf = Vec::new();
        {
            let mut enc = GifEncoder::new(&mut buf);
            enc.encode_frame(solid(image::Rgba([1, 2, 3, 255]), 100))
                .unwrap();
        }
        let result =
            convert_gif_buffer_to_anaglyph(&buf, VideoDirection::Clockwise, AnaglyphType::Color);
        assert!(matches!(result, Err(e) if e.contains("at least two frames")));
    }
}
