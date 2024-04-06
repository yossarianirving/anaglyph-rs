use std::path::Path;
use ndarray::{ArrayBase, Dim, OwnedRepr};

use video_rs::decode::Decoder;
use video_rs::encode::{Encoder, Settings};

use rayon::prelude::*;

use crate::anaglyph;

pub enum VideoDirection {
    Clockwise,
    CounterClockwise,
}


pub fn convert_video_to_anaglyph(video: &str, video_out: &str, direction: VideoDirection) {
    video_rs::init().unwrap();
    let mut decoder = match Decoder::new(Path::new(video)){
        Ok(decoder) => decoder,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };

    let video_size = decoder.size();
    let frame_rate = decoder.frame_rate();
    let encoder_settings = Settings::preset_h264_yuv420p(video_size.0 as usize, video_size.1 as usize, false);
    let mut encoder = match Encoder::new(Path::new(video_out), encoder_settings){
        Ok(encoder) => encoder,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };

    println!(" {:?}", video_size);

    let mut previous_frame = decoder.decode().unwrap();
    let mut new_frame = ArrayBase::<OwnedRepr<u8>, Dim<[usize; 3]>>::zeros((video_size.1 as usize, video_size.0 as usize, 3));
    for frame in decoder.decode_iter() {
        if let Ok((time, frame)) = frame {
            
            for i in 0..video_size.0 {
                for j in 0..video_size.1 {
                    let i = i as i32;
                    let j = j as i32;
                    let (left_slice, right_slice) = match direction {
                        VideoDirection::Clockwise => {
                            let left = frame.slice(ndarray::s![j, i, ..]).to_slice().unwrap();
                            let right = previous_frame.1.slice(ndarray::s![j, i, ..]).to_slice().unwrap();
                            (left, right)
                        },
                        VideoDirection::CounterClockwise => {
                            let left = previous_frame.1.slice(ndarray::s![j, i, ..]).to_slice().unwrap();
                            let right = frame.slice(ndarray::s![j, i, ..]).to_slice().unwrap();
                            (left, right)
                        }
                    };
                    let mut new_slice = new_frame.slice_mut(ndarray::s![j, i, ..]).into_slice().unwrap();
                    anaglyph::combine_slices(left_slice, right_slice, new_slice, &anaglyph::AnaglyphType::Color)
                }
            }
            encoder.encode(&new_frame, &previous_frame.0);
            previous_frame = (time, frame);
        } else {
            break
        }
    }
    
    encoder.finish().expect("failed to finish encoder");
    
}
