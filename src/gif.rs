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
/// Native builds with the `gpu` feature blend on the GPU (falling back to
/// the CPU when no device is available); other targets blend on the CPU.
/// On wasm, prefer [`convert_gif_buffer_to_anaglyph_async`], which can use
/// WebGPU without blocking the browser's event loop.
pub fn convert_gif_buffer_to_anaglyph(
    gif_data: &[u8],
    direction: VideoDirection,
    anaglyph_type: AnaglyphType,
) -> Vec<u8> {
    let cursor_in = Cursor::new(gif_data);
    let decoder = GifDecoder::new(cursor_in).unwrap();
    let gif_size = decoder.dimensions();
    let mut frames = decoder.into_frames();

    let anaglyph_matrix = anaglyph_type_to_matrix(anaglyph_type);

    let first_frame = frames
        .next()
        .expect("Failed to get first frame")
        .expect("Failed to get first frame");
    let delay = first_frame.delay();

    let mut previous_frame = first_frame.into_buffer(); // RgbaImage

    // GPU blending when available (native builds with the `gpu` feature).
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    let mut gpu = match GpuBlender::new(gif_size.0, gif_size.1, anaglyph_matrix) {
        Ok(blender) => {
            println!("Using GPU blending");
            Some(blender)
        }
        Err(e) => {
            eprintln!("GPU blending unavailable ({}), falling back to CPU", e);
            None
        }
    };

    let mut new_frames = Vec::<Frame>::new();
    for frame in frames {
        let frame = frame.expect("Failed to get frame");
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
            Some(blender) => blender.blend_rgba(left, right),
            None => combine_rgba(left, right, anaglyph_matrix),
        };
        #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
        let blended = combine_rgba(left, right, anaglyph_matrix);

        let new_rgba_frame = RgbaImage::from_vec(gif_size.0, gif_size.1, blended).unwrap();
        new_frames.push(Frame::from_parts(new_rgba_frame, 0, 0, delay));
        previous_frame = rgba_frame;
    }

    encode_anaglyph(new_frames)
}

/// Async variant of [`convert_gif_buffer_to_anaglyph`] for wasm: initializes
/// the WebGPU device and blends frames without blocking the event loop,
/// falling back to CPU blending when WebGPU is unavailable.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub async fn convert_gif_buffer_to_anaglyph_async(
    gif_data: &[u8],
    direction: VideoDirection,
    anaglyph_type: AnaglyphType,
) -> Result<Vec<u8>, String> {
    let cursor_in = Cursor::new(gif_data);
    let decoder =
        GifDecoder::new(cursor_in).map_err(|e| format!("failed to decode gif: {e}"))?;
    let gif_size = decoder.dimensions();
    let mut frames = decoder.into_frames();

    let anaglyph_matrix = anaglyph_type_to_matrix(anaglyph_type);

    let first_frame = frames
        .next()
        .ok_or("gif contains no frames")?
        .map_err(|e| format!("failed to decode frame: {e}"))?;
    let delay = first_frame.delay();
    let mut previous_frame = first_frame.into_buffer();

    // Fall back to CPU blending when WebGPU is unavailable; report the
    // reason on the JS console so the fallback is visible during debugging.
    let mut gpu = match GpuBlender::new_async(gif_size.0, gif_size.1, anaglyph_matrix).await {
        Ok(blender) => Some(blender),
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
        let rgba_frame = frame
            .map_err(|e| format!("failed to decode frame: {e}"))?
            .into_buffer();

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

        let blended = match &mut gpu {
            Some(blender) => blender.blend_rgba_async(left, right).await?,
            None => combine_rgba(left, right, anaglyph_matrix),
        };

        let new_rgba_frame = RgbaImage::from_vec(gif_size.0, gif_size.1, blended)
            .ok_or("blended frame has unexpected size")?;
        new_frames.push(Frame::from_parts(new_rgba_frame, 0, 0, delay));
        previous_frame = rgba_frame;
    }

    Ok(encode_anaglyph(new_frames))
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
fn combine_rgba_pixel(
    left: &[u8],
    right: &[u8],
    out: &mut [u8],
    anaglyph_matrix: &[[f32; 9]; 2],
) {
    if left[3] == 0 && right[3] == 0 {
        return; // left as [0, 0, 0, 0]: fully transparent
    }
    combine_slices(&left[..3], &right[..3], &mut out[..3], anaglyph_matrix);
    out[3] = 255;
}

/// Mux blended frames into an animated gif, looping forever.
fn encode_anaglyph(new_frames: Vec<Frame>) -> Vec<u8> {
    let mut out_buffer = Vec::new();
    {
        let mut encoder = GifEncoder::new_with_speed(&mut out_buffer, 10);
        encoder
            .set_repeat(image::codecs::gif::Repeat::Infinite)
            .expect("Failed to set repeat");
        encoder
            .encode_frames(new_frames)
            .expect("Failed to encode frames");
    }
    out_buffer
}

/// Wrapper function for file-based usage.
/// Retains the original signature and calls the buffer-based function.
pub fn convert_gif_to_anaglyph(
    gif: &str,
    gif_out: &str,
    direction: VideoDirection,
    anaglyph_type: AnaglyphType,
) {
    // Read input file into buffer
    let mut input_file = File::open(gif).expect("File not found");
    let mut gif_data = Vec::new();
    input_file
        .read_to_end(&mut gif_data)
        .expect("Failed to read file");

    // Call buffer-based function
    let out_data = convert_gif_buffer_to_anaglyph(&gif_data, direction, anaglyph_type);

    // Write output buffer to file
    let mut output_file = File::create(gif_out).expect("Failed to create file");
    output_file
        .write_all(&out_data)
        .expect("Failed to write file");
}
