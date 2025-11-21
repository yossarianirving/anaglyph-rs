use crate::anaglyph::{anaglyph_type_to_matrix, combine_slices, AnaglyphType, VideoDirection};
use image::buffer::ConvertBuffer;
use image::codecs::gif::{GifDecoder, GifEncoder};
use image::{AnimationDecoder, DynamicImage, Frame, ImageDecoder, RgbImage};
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

    let mut previous_frame = DynamicImage::ImageRgba8(first_frame.into_buffer())
        .into_rgb8()
        .into_vec();

    let mut new_frames = Vec::<Frame>::new();
    for (_frame_index, frame) in frames.enumerate() {
        let frame = frame.expect("Failed to get frame");
        let frame = ConvertBuffer::<RgbImage>::convert(&frame.into_buffer()).into_vec();
        let mut new_frame = vec![255; (gif_size.0 * gif_size.1 * 3) as usize];
        for i in 0..gif_size.0 {
            for j in 0..gif_size.1 {
                let index = (j * gif_size.0 + i) as usize * 3;
                let (left_slice, right_slice) = match direction {
                    VideoDirection::Clockwise => {
                        let left = &frame[index..index + 3];
                        let right = &previous_frame[index..index + 3];
                        (left, right)
                    }
                    VideoDirection::CounterClockwise => {
                        let left = &previous_frame[index..index + 3];
                        let right = &frame[index..index + 3];
                        (left, right)
                    }
                };
                let mut anaglyph_slice = vec![0, 0, 0];
                combine_slices(
                    &left_slice,
                    &right_slice,
                    &mut anaglyph_slice,
                    &anaglyph_matrix,
                );
                new_frame[index..index + 3].clone_from_slice(&anaglyph_slice);
            }
        }
        let new_rgb_frame = RgbImage::from_vec(gif_size.0, gif_size.1, new_frame).unwrap();
        new_frames.push(Frame::from_parts(
            DynamicImage::ImageRgb8(new_rgb_frame).into_rgba8(),
            0,
            0,
            delay,
        ));
        previous_frame = frame;
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
