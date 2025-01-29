use image::codecs::gif::{GifDecoder, GifEncoder};
use image::{AnimationDecoder, DynamicImage, Frame, ImageDecoder, RgbImage};
use image::buffer::ConvertBuffer;
use std::fs::File;
use std::io::BufReader;
use crate::anaglyph::{VideoDirection, AnaglyphType, anaglyph_type_to_matrix, combine_slices};

pub fn convert_gif_to_anaglyph(gif: &str, gif_out: &str, direction: VideoDirection, anaglyph_type: AnaglyphType) {
    let file_in = BufReader::new(File::open(gif).expect("File not found"));
    let mut decoder = GifDecoder::new(file_in).unwrap();
    let gif_size = decoder.dimensions();
    let mut frames = decoder.into_frames();

    let anaglyph_matrix = anaglyph_type_to_matrix(anaglyph_type);

    let first_frame = frames.next().expect("Failed to get first frame").expect("Failed to get first frame");
    let delay = first_frame.delay();

    let mut previous_frame = DynamicImage::ImageRgba8(first_frame.into_buffer()).into_rgb8().into_vec();
    
    let mut new_frames = Vec::<Frame>::new();
    for (frame_index, frame) in frames.enumerate() {
        let frame = frame.expect("Failed to get frame");
        let frame = ConvertBuffer::<RgbImage>::convert(&frame.into_buffer()).into_vec();
        let mut new_frame = vec![255; (gif_size.0 * gif_size.1 * 3) as usize];
        for i in 0..gif_size.0 {
            for j in 0..gif_size.1 {
                let index = (j * gif_size.0 + i ) as usize * 3;
                let (left_slice, right_slice) = match direction {
                    VideoDirection::Clockwise => {
                        let left = &frame[index..index+3];
                        let right = &previous_frame[index..index+3];
                        (left, right)
                    }
                    VideoDirection::CounterClockwise => {
                        let left = &previous_frame[index..index+3];
                        let right = &frame[index..index+3];
                        (left, right)
                    }
                };
                let mut anaglyph_slice = vec![0, 0, 0];
                combine_slices(&left_slice, &right_slice, &mut anaglyph_slice, &anaglyph_matrix);
                new_frame[index..index+3].clone_from_slice(&anaglyph_slice);
            }
        }
        let new_rgb_frame = RgbImage::from_vec(gif_size.0, gif_size.1, new_frame).unwrap();
        new_frames.push(Frame::from_parts(DynamicImage::ImageRgb8(new_rgb_frame).into_rgba8(), 0, 0, delay));
        previous_frame = frame;
        println!("Processed frame {}", frame_index + 1);
    }

    println!("Total frames processed: {}", new_frames.len());
    if !std::path::Path::new(gif_out).exists() {
        File::create(gif_out).expect("Failed to create file");
    }
    
    let file_out = File::options().write(true).open(gif_out).expect("Failed to open file");
    let mut encoder = GifEncoder::new_with_speed(file_out, 10);
    encoder.set_repeat(image::codecs::gif::Repeat::Infinite).expect("Failed to set repeat");
    encoder.encode_frames(new_frames).expect("Failed to encode frames");

    println!("GIF encoding completed successfully");
}