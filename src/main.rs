use anaglyph_rs::anaglyph::{
    left_right_to_anaglyph, left_right_to_anaglyph_offset, AnaglyphType, Offset, VideoDirection,
};
use anaglyph_rs::gif::convert_gif_to_anaglyph;
#[cfg(feature = "video")]
use anaglyph_rs::video;
use clap::{arg, command, value_parser};
use image::{imageops, DynamicImage, ImageReader, RgbImage};

fn main() {
    let matches = command!()
        .arg(arg!(-l --left <FILE> "Left image").required(false))
        .arg(arg!(-r --right <FILE> "Right image").required(false))
        .arg(arg!(-s --stereo <FILE> "Stereoscopic image").required(false))
        .arg(arg!(-o --out <FILE> "Output file").required(true))
        .arg(arg!(-t --type <TYPE> "Type of anaglyph").required(false).value_parser(["color", "half-color", "grayscale", "optimized", "true"]).default_value("color"))
        .arg(arg!(-x --"offset-x" <OFFSET> "Offset in x direction (images only)").required(false).allow_negative_numbers(true).value_parser(value_parser!(i32)).default_value("0"))
        .arg(arg!(-y --"offset-y" <OFFSET> "Offset in y direction (images only)").required(false).allow_negative_numbers(true).value_parser(value_parser!(i32)).default_value("0"))
        .arg(arg!(-v --video <FILE> "Video file").required(false))
        .arg(arg!(-g --gif <FILE> "Gif file").required(false))
        .arg(arg!(-d --"video-direction" <DIRECTION> "Direction of video/gif (clockwise or counter-clockwise)").required(false).value_parser(["clockwise", "counter-clockwise"]).default_value("clockwise"))
        .get_matches();

    let anaglyph_type = match matches
        .get_one::<String>("type")
        .expect("Type should not be empty")
        .as_str()
    {
        "color" => AnaglyphType::Color,
        "half-color" => AnaglyphType::HalfColor,
        "grayscale" => AnaglyphType::GrayScale,
        "optimized" => AnaglyphType::Optimized,
        "true" => AnaglyphType::True,
        _ => panic!("Invalid anaglyph type"),
    };
    let offset = Offset {
        x: matches.get_one::<i32>("offset-x").unwrap().clone(),
        y: matches.get_one::<i32>("offset-y").unwrap().clone(),
    };
    let (left, right, stereo, video, gif) = (
        matches.get_one::<String>("left"),
        matches.get_one::<String>("right"),
        matches.get_one::<String>("stereo"),
        matches.get_one::<String>("video"),
        matches.get_one::<String>("gif"),
    );

    let output = matches
        .get_one::<String>("out")
        .expect("Output should not be empty");

    match (left, right, stereo, video, gif) {
        (Some(l), Some(r), None, None, None) => {
            let anaglyph = convert_left_right(l, r, anaglyph_type, Some(offset));
            match anaglyph.save(output) {
                Ok(_) => println!(""),
                Err(i) => panic!("{}", i),
            };
        }
        (None, None, Some(s), None, None) => {
            let anaglyph = convert_stereoscopic(s.to_string(), anaglyph_type, Some(offset));
            match anaglyph.save(output) {
                Ok(_) => println!(""),
                Err(i) => panic!("{}", i),
            };
        }
        #[cfg(feature = "video")]
        (None, None, None, Some(v), None) => {
            let direction = match matches
                .get_one::<String>("video-direction")
                .expect("Direction should not be empty")
                .as_str()
            {
                "clockwise" => VideoDirection::Clockwise,
                "counter-clockwise" => VideoDirection::CounterClockwise,
                _ => panic!("Invalid video direction"),
            };
            convert_video_to_anaglyph(v, output, direction, anaglyph_type);
        }
        (None, None, None, None, Some(g)) => {
            let direction = match matches
                .get_one::<String>("video-direction")
                .expect("Direction should not be empty")
                .as_str()
            {
                "clockwise" => VideoDirection::Clockwise,
                "counter-clockwise" => VideoDirection::CounterClockwise,
                _ => panic!("Invalid video direction"),
            };
            convert_gif_to_anaglyph(g, output, direction, anaglyph_type);
        }
        #[cfg(not(feature = "video"))]
        (_, _, _, Some(_), None) => panic!("Video support not enabled"),
        _ => panic!("No or invalid input provided"),
    }
}

// convert left/right into anaglpyh
fn convert_left_right(
    left: &String,
    right: &String,
    anaglyph_type: AnaglyphType,
    offset: Option<Offset>,
) -> DynamicImage {
    let left_image: RgbImage = match ImageReader::open(left) {
        Ok(r) => r.decode().unwrap().into_rgb8(), // clunky
        Err(e) => panic!("{}", e),
    };

    let right_image: RgbImage = match ImageReader::open(right) {
        Ok(r) => r.decode().unwrap().into_rgb8(),
        Err(e) => panic!("{}", e),
    };

    if left_image.height() != right_image.height() || left_image.width() != right_image.width() {
        panic!("Left and right images are not the same size");
    }

    match offset {
        Some(i) => image::DynamicImage::ImageRgb8(left_right_to_anaglyph_offset(
            &left_image,
            &right_image,
            anaglyph_type,
            i,
        )),
        None => image::DynamicImage::ImageRgb8(left_right_to_anaglyph(
            &left_image,
            &right_image,
            anaglyph_type,
        )),
    }
}
// convert stereoscopic into anaglyph
fn convert_stereoscopic(
    stereoscopic: String,
    anaglyph_type: AnaglyphType,
    offset: Option<Offset>,
) -> DynamicImage {
    let stereoscopic_image: RgbImage = match ImageReader::open(stereoscopic) {
        Ok(r) => r.decode().unwrap().into_rgb8(),
        Err(e) => panic!("{}", e),
    };
    let width = match stereoscopic_image.width() {
        w if w % 2 == 0 => w / 2,
        _ => panic!("Stereoscopic image must have an even width"),
    };
    let height = stereoscopic_image.height();

    let left_image = imageops::crop_imm(&stereoscopic_image, 0, 0, width, height).to_image();

    let right_image = imageops::crop_imm(&stereoscopic_image, width, 0, width, height).to_image();

    match offset {
        Some(i) => image::DynamicImage::ImageRgb8(left_right_to_anaglyph_offset(
            &left_image,
            &right_image,
            anaglyph_type,
            i,
        )),
        None => image::DynamicImage::ImageRgb8(left_right_to_anaglyph(
            &left_image,
            &right_image,
            anaglyph_type,
        )),
    }
}

#[cfg(feature = "video")]
fn convert_video_to_anaglyph(
    video: &str,
    video_out: &str,
    direction: VideoDirection,
    anaglyph_type: AnaglyphType,
) {
    video::convert_video_to_anaglyph(video, video_out, direction, anaglyph_type);
}
