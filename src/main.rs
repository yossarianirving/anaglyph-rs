use anaglyph_rs::anaglyph::{left_right_to_anaglyph_offset, AnaglyphType, Offset, VideoDirection};
use anaglyph_rs::gif::convert_gif_to_anaglyph;
#[cfg(feature = "video")]
use anaglyph_rs::video;
use clap::{arg, command, value_parser};
use image::{imageops, DynamicImage, ImageReader, RgbImage};
use std::str::FromStr;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let matches = command!()
        .arg(arg!(-l --left <FILE> "Left image").required(false).requires("right"))
        .arg(arg!(-r --right <FILE> "Right image").required(false).requires("left"))
        .arg(arg!(-s --stereo <FILE> "Stereoscopic image").required(false).conflicts_with_all(["left", "right"]))
        .arg(arg!(-o --out <FILE> "Output file").required(true))
        .arg(arg!(-t --type <TYPE> "Type of anaglyph").required(false).value_parser(["color", "half-color", "grayscale", "optimized", "true"]).default_value("color"))
        .arg(arg!(-x --"offset-x" <OFFSET> "Offset in x direction (images only)").required(false).allow_negative_numbers(true).value_parser(value_parser!(i32)).default_value("0").conflicts_with_all(["video", "gif"]))
        .arg(arg!(-y --"offset-y" <OFFSET> "Offset in y direction (images only)").required(false).allow_negative_numbers(true).value_parser(value_parser!(i32)).default_value("0").conflicts_with_all(["video", "gif"]))
        .arg(arg!(-v --video <FILE> "Video file").required(false).conflicts_with_all(["left", "right", "stereo", "gif"]))
        .arg(arg!(-g --gif <FILE> "Gif file").required(false).conflicts_with_all(["left", "right", "stereo", "video"]))
        .arg(arg!(-d --"video-direction" <DIRECTION> "Direction of video/gif (clockwise or counter-clockwise)").required(false).value_parser(["clockwise", "counter-clockwise"]).default_value("clockwise"))
        .arg(arg!(-e --encoder <ENCODER> "Video encoder backend (auto, hardware, or software)").required(false).value_parser(["auto", "hardware", "software"]).default_value("auto"))
        .arg(arg!(-D --decode <DECODER> "Video decoder mode (cpu or hardware)").required(false).value_parser(["cpu", "hardware"]).default_value("cpu"))
        .get_matches();

    // clap already restricts these to the values the FromStr impls accept.
    let anaglyph_type = AnaglyphType::from_str(
        matches
            .get_one::<String>("type")
            .expect("clap provides a default type"),
    )
    .expect("clap restricts --type to valid values");
    let direction = VideoDirection::from_str(
        matches
            .get_one::<String>("video-direction")
            .expect("clap provides a default direction"),
    )
    .expect("clap restricts --video-direction to valid values");
    let offset = Offset {
        x: *matches.get_one::<i32>("offset-x").expect("clap default"),
        y: *matches.get_one::<i32>("offset-y").expect("clap default"),
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
        .expect("output is required");

    match (left, right, stereo, video, gif) {
        (Some(l), Some(r), None, None, None) => {
            let anaglyph = convert_left_right(l, r, anaglyph_type, offset)?;
            save_image(&anaglyph, output)
        }
        (None, None, Some(s), None, None) => {
            let anaglyph = convert_stereoscopic(s, anaglyph_type, offset)?;
            save_image(&anaglyph, output)
        }
        #[cfg(feature = "video")]
        (None, None, None, Some(v), None) => {
            let encoder = match matches
                .get_one::<String>("encoder")
                .expect("clap provides a default encoder")
                .as_str()
            {
                "hardware" => video::EncoderChoice::Hardware,
                "software" => video::EncoderChoice::Software,
                _ => video::EncoderChoice::Auto,
            };
            let decoder = match matches
                .get_one::<String>("decode")
                .expect("clap provides a default decoder mode")
                .as_str()
            {
                "hardware" => video::DecoderChoice::Hardware,
                _ => video::DecoderChoice::Cpu,
            };
            let report = video::convert_video_to_anaglyph(
                v,
                output,
                direction,
                anaglyph_type,
                decoder,
                encoder,
            )?;
            println!("decoder: {}", report.decoder);
            println!("blend:   {}", report.blender);
            println!("encoder: {}", report.encoder);
            println!("encoded {} frames", report.frames);
            Ok(())
        }
        (None, None, None, None, Some(g)) => {
            convert_gif_to_anaglyph(g, output, direction, anaglyph_type)
        }
        #[cfg(not(feature = "video"))]
        (_, _, _, Some(_), None) => {
            Err("video support not enabled (build with `--features video`)".into())
        }
        _ => Err("no input provided: use --left/--right, --stereo, --video or --gif".into()),
    }
}

// convert left/right into anaglyph
fn convert_left_right(
    left: &str,
    right: &str,
    anaglyph_type: AnaglyphType,
    offset: Offset,
) -> Result<DynamicImage, String> {
    let left_image = load_rgb(left)?;
    let right_image = load_rgb(right)?;

    left_right_to_anaglyph_offset(&left_image, &right_image, anaglyph_type, offset)
        .map(DynamicImage::ImageRgb8)
}

// convert stereoscopic into anaglyph
fn convert_stereoscopic(
    stereoscopic: &str,
    anaglyph_type: AnaglyphType,
    offset: Offset,
) -> Result<DynamicImage, String> {
    let stereoscopic_image = load_rgb(stereoscopic)?;
    let width = match stereoscopic_image.width() {
        w if w % 2 == 0 => w / 2,
        _ => return Err("stereoscopic image must have an even width".into()),
    };
    let height = stereoscopic_image.height();

    let left_image = imageops::crop_imm(&stereoscopic_image, 0, 0, width, height).to_image();
    let right_image = imageops::crop_imm(&stereoscopic_image, width, 0, width, height).to_image();

    left_right_to_anaglyph_offset(&left_image, &right_image, anaglyph_type, offset)
        .map(DynamicImage::ImageRgb8)
}

/// Decode an image file as RGB8.
fn load_rgb(path: &str) -> Result<RgbImage, String> {
    ImageReader::open(path)
        .map_err(|e| format!("failed to open {path}: {e}"))?
        .decode()
        .map_err(|e| format!("failed to decode {path}: {e}"))
        .map(DynamicImage::into_rgb8)
}

/// Save an image, with the format inferred from the output extension.
fn save_image(image: &DynamicImage, path: &str) -> Result<(), String> {
    image
        .save(path)
        .map_err(|e| format!("failed to save {path}: {e}"))
}
