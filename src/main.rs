
use anaglyph_rs::{anaglyph::{left_right_to_anaglyph, left_right_to_anaglyph_offset, AnaglyphType, Offset}, video};
use clap::Parser;
use image::{imageops, io::Reader as ImageReader, DynamicImage, RgbImage};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    left: Option<String>,
    #[arg(short, long)]
    right: Option<String>,
    #[arg(short, long)]
    stereo: Option<String>,
    #[arg(short, long)]
    out: Option<String>,
    #[arg(short)]
    t: Option<String>,
    #[arg(long)]
    offset_x: Option<i32>,
    #[arg(long)]
    offset_y: Option<i32>,
    #[arg(long)]
    video: Option<String>,
    #[arg(long)]
    video_direction: Option<String>,
    #[arg(long)]
    video_out: Option<String>
}

fn main() {
    let args = Args::parse();
    let anaglyph_type = match args.t.as_deref() {
        Some("color") => AnaglyphType::Color,
        Some("half-color") => AnaglyphType::HalfColor,
        Some("grayscale") => AnaglyphType::GrayScale,
        Some("optimized") => AnaglyphType::Optimized,
        Some("true") => AnaglyphType::True,
        Some(_) | None => AnaglyphType::Color

    };
    // let offset = match (args.offset_x, args.offset_y) {
    //     (Some(x), Some(y)) => Some(Offset { x , y }),
    //     _ => None
    // };
    // let anaglyph: DynamicImage = match args {
    //     Args { left: Some(left), right: Some(right), ..} => convert_left_right(left, right, anaglyph_type, offset),
    //     Args { stereo: Some(stereo), ..} => convert_stereoscopic(stereo, anaglyph_type, offset),
    //     Args {..} => {
    //         println!("Nothing!!!");
    //         DynamicImage::new(0, 0, image::ColorType::Rgb8)
    //     } 
    // };


    // let output_name = match args.out {
    //     Some(name) => name,
    //     None => "output.jpg".to_owned()
    // };
    // match anaglyph.save(output_name) {
    //     Ok(_) => println!(""),
    //     Err(i) => panic!("{}", i)
    // };
    video::convert_video_to_anaglyph(args.video.as_ref().unwrap(), args.video_out.as_ref().unwrap(), video::VideoDirection::Clockwise);
}

// convert left/right into anaglpyh
fn convert_left_right(left: String, right: String, anaglyph_type: AnaglyphType, offset: Option<Offset>) -> DynamicImage {
    let left_image: RgbImage = match ImageReader::open(left) {
        Ok(r) => r.decode().unwrap().into_rgb8(), // clunky
        Err(e) => panic!("{}", e)
    };

    let right_image: RgbImage = match ImageReader::open(right) {
        Ok(r) => r.decode().unwrap().into_rgb8(),
        Err(e) => panic!("{}", e)
    };

    if left_image.height() != right_image.height() || left_image.width() != right_image.width() {
        panic!("Left and right images are not the same size");
    }

    match offset {
        Some(i) => image::DynamicImage::ImageRgb8(left_right_to_anaglyph_offset(&left_image, &right_image, anaglyph_type, i)),
        None => image::DynamicImage::ImageRgb8(left_right_to_anaglyph(&left_image, &right_image, anaglyph_type))
    }
}
// convert stereoscopic into anaglyph
fn convert_stereoscopic(stereoscopic: String, anaglyph_type: AnaglyphType, offset: Option<Offset>) -> DynamicImage {
    let stereoscopic_image: RgbImage = match ImageReader::open(stereoscopic) {
        Ok(r) => r.decode().unwrap().into_rgb8(),
        Err(e) => panic!("{}", e)
    };
    let width = match stereoscopic_image.width() {
        w if w % 2 == 0 => w / 2,
        _ => panic!("Stereoscopic image must have an even width")
    };
    let height = stereoscopic_image.height();

    let left_image = imageops::crop_imm(&stereoscopic_image, 0, 0, width, height).to_image();

    let right_image = imageops::crop_imm(&stereoscopic_image, width, 0, width, height).to_image();


    match offset {
        Some(i) => image::DynamicImage::ImageRgb8(left_right_to_anaglyph_offset(&left_image, &right_image, anaglyph_type, i)),
        None => image::DynamicImage::ImageRgb8(left_right_to_anaglyph(&left_image, &right_image, anaglyph_type))
    }
}