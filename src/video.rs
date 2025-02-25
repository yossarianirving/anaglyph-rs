use ndarray::{ArrayBase, Dim, OwnedRepr};
use std::path::Path;

use video_rs::decode::Decoder;
use video_rs::encode::{Encoder, Settings};

use std::sync::mpsc;
use std::thread;

use crate::anaglyph::{VideoDirection, AnaglyphType, anaglyph_type_to_matrix, combine_slices};

pub fn convert_video_to_anaglyph(video: &str, video_out: &str, direction: VideoDirection, anaglyph_type: AnaglyphType) {
    video_rs::init().unwrap();
    let mut decoder = match Decoder::new(Path::new(video)) {
        Ok(decoder) => decoder,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };


    let video_size = decoder.size();
    let encoder_settings =
        Settings::preset_h264_yuv420p(video_size.0 as usize, video_size.1 as usize, false);
    let mut encoder = match Encoder::new(Path::new(video_out), encoder_settings) {
        Ok(encoder) => encoder,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };

    println!(" {:?}", video_size);
    let (tx, rx) = mpsc::channel();

    let mut previous_frame = decoder.decode().unwrap();

    let anaglyph_matrix = anaglyph_type_to_matrix(anaglyph_type);

    thread::spawn(move || {
        for frame in decoder.decode_iter() {
            if let Ok((time, frame)) = frame {
                let mut new_frame = ArrayBase::<OwnedRepr<u8>, Dim<[usize; 3]>>::zeros((
                    video_size.1 as usize,
                    video_size.0 as usize,
                    3,
                ));
                for i in 0..video_size.0 {
                    for j in 0..video_size.1 {
                        let i = i as i32;
                        let j = j as i32;
                        let (left_slice, right_slice) = match direction {
                            VideoDirection::Clockwise => {
                                let left = frame.slice(ndarray::s![j, i, ..]).to_slice().unwrap();
                                let right = previous_frame
                                    .1
                                    .slice(ndarray::s![j, i, ..])
                                    .to_slice()
                                    .unwrap();
                                (left, right)
                            }
                            VideoDirection::CounterClockwise => {
                                let left = previous_frame
                                    .1
                                    .slice(ndarray::s![j, i, ..])
                                    .to_slice()
                                    .unwrap();
                                let right = frame.slice(ndarray::s![j, i, ..]).to_slice().unwrap();
                                (left, right)
                            }
                        };
                        let new_slice = new_frame
                            .slice_mut(ndarray::s![j, i, ..])
                            .into_slice()
                            .unwrap();
                        combine_slices(
                            left_slice,
                            right_slice,
                            new_slice,
                            anaglyph_matrix,
                        )
                    }
                }

                // encoder.encode(&new_frame, &previous_frame.0);
                tx.send((previous_frame.0.clone(), new_frame)).expect(":(");
                previous_frame = (time, frame);
            } else {
                break;
            }
        }
    });

    for recieved in rx {
        let _ = encoder.encode(&recieved.1, &recieved.0);
    }

    encoder.finish().expect("failed to finish encoder");
}
