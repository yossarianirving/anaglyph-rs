use anaglyph_rs::anaglyph::{
    left_right_to_anaglyph_offset, AnaglyphType, BlenderChoice, Offset, VideoDirection,
};
use anaglyph_rs::gif::convert_gif_to_anaglyph;
#[cfg(feature = "video")]
use anaglyph_rs::video;
use clap::{arg, command, value_parser};
use image::{imageops, DynamicImage, ImageReader, RgbImage};
use std::path::Path;
use std::str::FromStr;

fn main() {
    // Safety net: a panic (ours or a dependency's, e.g. deep inside the
    // FFmpeg bindings on a pathological file) should still exit as a
    // reported error rather than a crash. Panics on worker threads are
    // caught by `JoinHandle` and surfaced as errors instead; this covers
    // the main thread. The default panic hook has already printed the
    // panic location before this runs.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run));
    let outcome = match result {
        Ok(inner) => inner,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            // The default hook prints the panic site but not a backtrace by
            // default; point at the switch that gets one.
            eprintln!("hint: re-run with RUST_BACKTRACE=1 for details");
            Err(format!("unexpected internal error: {msg}"))
        }
    };
    if let Err(e) = outcome {
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
        .arg(arg!(-d --"video-direction" <DIRECTION> "Direction of video/gif; auto detects direction changes (clockwise, counter-clockwise, auto, auto-reverse)").required(false).value_parser(["clockwise", "counter-clockwise", "auto", "auto-reverse"]).default_value("clockwise"))
        .arg(arg!(--"motion-json" <FILE> "Write the detected motion timeline to a JSON file (requires --video with -d auto)").required(false))
        .arg(arg!(-e --encoder <ENCODER> "Video encoder backend (auto, hardware, or software)").required(false).value_parser(["auto", "hardware", "software"]).default_value("auto"))
        .arg(arg!(-D --decode <DECODER> "Video decoder mode (cpu or hardware)").required(false).value_parser(["cpu", "hardware"]).default_value("cpu"))
        .arg(arg!(-b --blend <BLEND> "Blend backend for video and gif (auto, gpu, or cpu); images always blend on the CPU").required(false).value_parser(["auto", "gpu", "cpu"]).default_value("auto"))
        .get_matches();

    // clap already restricts these to the values the FromStr impls accept.
    let anaglyph_type = AnaglyphType::from_str(
        matches
            .get_one::<String>("type")
            .expect("clap provides a default type"),
    )
    .expect("clap restricts --type to valid values");
    let direction_name = matches
        .get_one::<String>("video-direction")
        .expect("clap provides a default direction")
        .as_str();
    let auto_direction = match direction_name {
        "auto" => Some(false),
        "auto-reverse" => Some(true),
        _ => None,
    };
    let offset = Offset {
        x: *matches.get_one::<i32>("offset-x").expect("clap default"),
        y: *matches.get_one::<i32>("offset-y").expect("clap default"),
    };
    let blend = BlenderChoice::from_str(matches.get_one::<String>("blend").expect("clap default"))
        .expect("clap restricts --blend to valid values");

    let (left, right, stereo, video, gif) = (
        matches.get_one::<String>("left"),
        matches.get_one::<String>("right"),
        matches.get_one::<String>("stereo"),
        matches.get_one::<String>("video"),
        matches.get_one::<String>("gif"),
    );
    if auto_direction.is_some() && video.is_none() {
        return Err(
            "automatic direction (-d auto / auto-reverse) is only supported with --video".into(),
        );
    }
    if blend == BlenderChoice::Gpu && (left.is_some() || stereo.is_some()) {
        return Err(
            "--blend gpu applies only to --video and --gif; images are always blended on the CPU"
                .into(),
        );
    }

    let output = matches
        .get_one::<String>("out")
        .expect("output is required");
    // Fail before opening inputs and initializing the GPU when the output
    // simply cannot be written, with a message naming the real problem.
    if let Some(dir) = Path::new(output).parent() {
        if !dir.as_os_str().is_empty() && !dir.is_dir() {
            return Err(format!("output directory {} does not exist", dir.display()));
        }
    }

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
            let motion_json = matches.get_one::<String>("motion-json").map(String::as_str);
            if motion_json.is_some() && auto_direction.is_none() {
                return Err(
                    "--motion-json requires --video with -d auto or -d auto-reverse".into(),
                );
            }
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
            let direction = match auto_direction {
                Some(reverse) => video::DirectionChoice::Auto { reverse },
                None => video::DirectionChoice::Fixed(
                    VideoDirection::from_str(direction_name)
                        .expect("clap restricts --video-direction to valid values"),
                ),
            };
            let report = video::convert_video_to_anaglyph(
                v,
                output,
                direction,
                anaglyph_type,
                decoder,
                encoder,
                blend,
            )?;
            println!("decoder: {}", report.decoder);
            println!("blend:   {}", report.blender);
            println!("encoder: {}", report.encoder);
            for reversal in &report.reversals {
                println!(
                    "reversal at {:.2}s (frame {}): {} -> {}",
                    reversal.time,
                    reversal.frame,
                    reversal.from.name(),
                    reversal.to.name()
                );
            }
            println!("encoded {} frames", report.frames);
            if let Some(path) = motion_json {
                write_motion_json(path, &report)?;
                println!("motion timeline: {path}");
            }
            Ok(())
        }
        (None, None, None, None, Some(g)) => {
            let direction = VideoDirection::from_str(direction_name)
                .expect("clap restricts --video-direction to valid values");
            convert_gif_to_anaglyph(g, output, direction, anaglyph_type, blend)
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

/// Write the detected motion timeline as a JSON file. The schema is small and
/// fixed, so it is emitted directly rather than pulling in a serde
/// dependency for one report.
#[cfg(feature = "video")]
fn write_motion_json(path: &str, report: &video::ConversionReport) -> Result<(), String> {
    let reversals: Vec<String> = report
        .reversals
        .iter()
        .map(|r| {
            format!(
                "    {{ \"frame\": {}, \"time\": {:.3}, \"from\": \"{}\", \"to\": \"{}\" }}",
                r.frame,
                r.time,
                r.from.name(),
                r.to.name()
            )
        })
        .collect();
    let frames: Vec<String> = report
        .motion
        .iter()
        .map(|p| {
            let trend = match p.trend {
                Some(anaglyph_rs::motion::Trend::Increasing) => "\"increasing\"",
                Some(anaglyph_rs::motion::Trend::Decreasing) => "\"decreasing\"",
                None => "null",
            };
            format!(
                "    {{ \"frame\": {}, \"time\": {:.3}, \"dx\": {:.4}, \"reliable\": {}, \"trend\": {} }}",
                p.frame, p.time, p.dx, p.reliable, trend
            )
        })
        .collect();
    let json = format!(
        "{{\n  \"reversals\": {},\n  \"frames\": {}\n}}\n",
        json_array(&reversals),
        json_array(&frames)
    );
    std::fs::write(path, json).map_err(|e| format!("failed to write {path}: {e}"))
}

/// A JSON array of pre-rendered entries, one per line (or `[]` when empty).
#[cfg(feature = "video")]
fn json_array(entries: &[String]) -> String {
    if entries.is_empty() {
        "[]".to_string()
    } else {
        format!("[\n{}\n  ]", entries.join(",\n"))
    }
}
