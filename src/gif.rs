use crate::anaglyph::{anaglyph_type_to_matrix, combine_slices, AnaglyphType, VideoDirection};
use image::codecs::gif::{GifDecoder, GifEncoder};
use image::{AnimationDecoder, Frame, ImageDecoder, RgbaImage};
use std::fs::File;
use std::io::{Cursor, Read, Write};

/// Core buffer-based function for GIF anaglyph conversion.
/// This can be called from WASM or anywhere you have GIF data in memory.
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

    let mut new_frames = Vec::<Frame>::new();
    for (_frame_index, frame) in frames.enumerate() {
        let frame = frame.expect("Failed to get frame");
        let rgba_frame = frame.into_buffer(); // RgbaImage

        let mut new_frame = Vec::with_capacity((gif_size.0 * gif_size.1 * 4) as usize); // RGBA

        for j in 0..gif_size.1 {
            for i in 0..gif_size.0 {
                let (left_pixel, right_pixel) = match direction {
                    VideoDirection::Clockwise => (
                        rgba_frame.get_pixel(i, j).0,
                        previous_frame.get_pixel(i, j).0,
                    ),
                    VideoDirection::CounterClockwise => (
                        previous_frame.get_pixel(i, j).0,
                        rgba_frame.get_pixel(i, j).0,
                    ),
                };

                // If either pixel is fully transparent, set output transparent
                if left_pixel[3] == 0 && right_pixel[3] == 0 {
                    new_frame.extend_from_slice(&[0, 0, 0, 0]);
                } else {
                    let mut anaglyph_slice = vec![0, 0, 0];
                    combine_slices(
                        &left_pixel[0..3],
                        &right_pixel[0..3],
                        &mut anaglyph_slice,
                        &anaglyph_matrix,
                    );
                    new_frame.extend_from_slice(&[
                        anaglyph_slice[0],
                        anaglyph_slice[1],
                        anaglyph_slice[2],
                        255,
                    ]);
                }
            }
        }
        let new_rgba_frame = RgbaImage::from_vec(gif_size.0, gif_size.1, new_frame).unwrap();
        new_frames.push(Frame::from_parts(new_rgba_frame, 0, 0, delay));
        previous_frame = rgba_frame;
        // Optionally: println!("Processed frame {}", frame_index + 1);
    }

    // Optionally: println!("Total frames processed: {}", new_frames.len());

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
