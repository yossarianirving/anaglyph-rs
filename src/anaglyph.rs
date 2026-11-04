use image::RgbImage;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnaglyphType {
    True,
    GrayScale,
    Optimized,
    Color,
    HalfColor,
}

/// Wire/CLI names for the anaglyph types, shared by the CLI, wasm bindings
/// and [`FromStr`].
impl AnaglyphType {
    pub fn name(self) -> &'static str {
        match self {
            AnaglyphType::True => "true",
            AnaglyphType::GrayScale => "grayscale",
            AnaglyphType::Optimized => "optimized",
            AnaglyphType::Color => "color",
            AnaglyphType::HalfColor => "half-color",
        }
    }
}

impl FromStr for AnaglyphType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "color" => Ok(AnaglyphType::Color),
            "half-color" => Ok(AnaglyphType::HalfColor),
            "grayscale" => Ok(AnaglyphType::GrayScale),
            "optimized" => Ok(AnaglyphType::Optimized),
            "true" => Ok(AnaglyphType::True),
            other => Err(format!(
                "unknown anaglyph type: {other} (expected color, half-color, grayscale, optimized or true)"
            )),
        }
    }
}

/// Which side of a frame pair feeds the left eye.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoDirection {
    /// The newer (current) frame is the left eye, the older (previous) frame
    /// the right eye.
    Clockwise,
    /// The older (previous) frame is the left eye, the newer (current) frame
    /// the right eye.
    CounterClockwise,
}

impl VideoDirection {
    pub fn name(self) -> &'static str {
        match self {
            VideoDirection::Clockwise => "clockwise",
            VideoDirection::CounterClockwise => "counter-clockwise",
        }
    }
}

impl FromStr for VideoDirection {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "clockwise" => Ok(VideoDirection::Clockwise),
            "counter-clockwise" => Ok(VideoDirection::CounterClockwise),
            other => Err(format!(
                "unknown direction: {other} (expected clockwise or counter-clockwise)"
            )),
        }
    }
}

/// Which backend blends the eye pixels.
///
/// Applies to the animated inputs (video and gif), which run the blend per
/// frame; images are always blended on the CPU. Builds without the `gpu`
/// feature have no GPU path at all, so [`Gpu`](BlenderChoice::Gpu) fails
/// with an explanatory error there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlenderChoice {
    /// Prefer GPU (wgpu) blending, fall back to the CPU when unavailable.
    #[default]
    Auto,
    /// Require GPU (wgpu) blending; fail if unavailable.
    Gpu,
    /// Always blend on the CPU.
    Cpu,
}

impl BlenderChoice {
    pub fn name(self) -> &'static str {
        match self {
            BlenderChoice::Auto => "auto",
            BlenderChoice::Gpu => "gpu",
            BlenderChoice::Cpu => "cpu",
        }
    }
}

impl FromStr for BlenderChoice {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(BlenderChoice::Auto),
            "gpu" => Ok(BlenderChoice::Gpu),
            "cpu" => Ok(BlenderChoice::Cpu),
            other => Err(format!(
                "unknown blend mode: {other} (expected auto, gpu or cpu)"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Offset {
    pub x: i32,
    pub y: i32,
}

const TRUE_MATRIX: [[f32; 9]; 2] = [
    [0.299, 0.587, 0.114, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.299, 0.587, 0.114],
];
const GRAY_SCALE_MATRIX: [[f32; 9]; 2] = [
    [0.299, 0.587, 0.114, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114],
];
const OPTIMIZED_MATRIX: [[f32; 9]; 2] = [
    [0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
];
const COLOR_MATRIX: [[f32; 9]; 2] = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
];
const HALF_COLOR_MATRIX: [[f32; 9]; 2] = [
    [0.299, 0.587, 0.114, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
];

pub fn anaglyph_type_to_matrix(anaglyph_type: AnaglyphType) -> &'static [[f32; 9]; 2] {
    match anaglyph_type {
        AnaglyphType::True => &TRUE_MATRIX,
        AnaglyphType::GrayScale => &GRAY_SCALE_MATRIX,
        AnaglyphType::Optimized => &OPTIMIZED_MATRIX,
        AnaglyphType::Color => &COLOR_MATRIX,
        AnaglyphType::HalfColor => &HALF_COLOR_MATRIX,
    }
}

/// Blend `left` and `right` into an anaglyph, sampling `left` at `(0, 0)` and
/// `right` at `right_off`, over an output area of `out_width` x `out_height`.
///
/// The per-row slice arithmetic is in bounds by construction: callers verify
/// that the offsets stay inside both source images.
#[allow(clippy::too_many_arguments)]
fn blend_region(
    left: &RgbImage,
    right: &RgbImage,
    matrix: &[[f32; 9]; 2],
    left_off: (u32, u32),
    right_off: (u32, u32),
    out_width: u32,
    out_height: u32,
) -> RgbImage {
    let out_row = out_width as usize * 3;
    let mut buffer = vec![0u8; out_row * out_height as usize];

    let left_bytes = left.as_raw().as_slice();
    let right_bytes = right.as_raw().as_slice();
    let left_pitch = left.width() as usize * 3;
    let right_pitch = right.width() as usize * 3;

    // Raster order (row-major) keeps both source and destination linearly
    // cache friendly, unlike indexing pixels with x as the outer loop.
    for y in 0..out_height as usize {
        let lo = (y + left_off.1 as usize) * left_pitch + left_off.0 as usize * 3;
        let ro = (y + right_off.1 as usize) * right_pitch + right_off.0 as usize * 3;
        let left_row = &left_bytes[lo..lo + out_row];
        let right_row = &right_bytes[ro..ro + out_row];
        let out = &mut buffer[y * out_row..][..out_row];

        for ((out, left), right) in out
            .chunks_exact_mut(3)
            .zip(left_row.chunks_exact(3))
            .zip(right_row.chunks_exact(3))
        {
            combine_slices(left, right, out, matrix);
        }
    }

    // The buffer length is derived from the dimensions, so this cannot fail.
    RgbImage::from_raw(out_width, out_height, buffer).expect("buffer sized from dimensions")
}

fn check_same_size(left: &RgbImage, right: &RgbImage) -> Result<(), String> {
    if left.dimensions() != right.dimensions() {
        return Err(format!(
            "left and right images must be the same size ({}x{} vs {}x{})",
            left.width(),
            left.height(),
            right.width(),
            right.height()
        ));
    }
    Ok(())
}

/// Blend `left` and `right` into an anaglyph shifted by `offset`: the output
/// keeps the pixel overlap of the two images, so it is `|offset|` smaller in
/// each axis it is offset along.
pub fn left_right_to_anaglyph_offset(
    left_image: &RgbImage,
    right_image: &RgbImage,
    anaglyph_type: AnaglyphType,
    offset: Offset,
) -> Result<RgbImage, String> {
    check_same_size(left_image, right_image)?;

    let abs_x = offset.x.unsigned_abs();
    let abs_y = offset.y.unsigned_abs();
    if abs_x >= left_image.width() {
        return Err("X offset cannot be greater than or equal to the image width".into());
    }
    if abs_y >= left_image.height() {
        return Err("Y offset cannot be greater than or equal to the image height".into());
    }

    // A positive x offset shifts the left image right, i.e. the left image
    // keeps a left margin of `x` pixels and the right image none (and vice
    // versa for negative offsets).
    let left_off = (offset.x.max(0) as u32, offset.y.max(0) as u32);
    let right_off = (
        if offset.x < 0 { abs_x } else { 0 },
        if offset.y < 0 { abs_y } else { 0 },
    );

    Ok(blend_region(
        left_image,
        right_image,
        anaglyph_type_to_matrix(anaglyph_type),
        left_off,
        right_off,
        left_image.width() - abs_x,
        left_image.height() - abs_y,
    ))
}

pub fn left_right_to_anaglyph(
    left_image: &RgbImage,
    right_image: &RgbImage,
    anaglyph_type: AnaglyphType,
) -> Result<RgbImage, String> {
    check_same_size(left_image, right_image)?;
    Ok(blend_region(
        left_image,
        right_image,
        anaglyph_type_to_matrix(anaglyph_type),
        (0, 0),
        (0, 0),
        left_image.width(),
        left_image.height(),
    ))
}

/// Quantize a blend sum to a byte: round half away from zero (matching
/// WGSL's `round`) and saturate to `0..=255`, so the CPU reference and the
/// GPU compute shader produce identical output and no matrix coefficient can
/// make the result wrap around.
///
/// Implemented as `floor(v + 0.5)` on the positive branch instead of
/// `f32::round`: on baseline x86-64 (no SSE4.1) `round` lowers to a scalar
/// libm call, costing ~2x the whole blend loop, while the plain truncating
/// cast vectorizes. For `v > 0` the two are bit-identical, and `v <= 0`
/// clamps to 0 either way.
#[inline]
fn to_u8(v: f32) -> u8 {
    if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        // v in (0, 255) so v + 0.5 < 255.5 and the cast cannot overflow.
        (v + 0.5) as u8
    }
}

#[inline]
fn dot3(px: [f32; 3], a: f32, b: f32, c: f32) -> f32 {
    px[0] * a + px[1] * b + px[2] * c
}

/// Blend one RGB pixel pair with the anaglyph matrix:
/// `out[c] = dot(M_left[c], left) + dot(M_right[c], right)`.
///
/// `left`, `right` and `anaglyph` are read as their first three bytes.
/// Fully unrolled with fixed indices: a dynamically indexed row loop defeats
/// the optimizer and costs ~2x here. `#[inline]` matters just as much — the
/// out-of-line form turns every pixel in the blend loops into an indirect
/// call and blocks vectorization.
#[inline]
pub fn combine_slices(left: &[u8], right: &[u8], anaglyph: &mut [u8], m: &[[f32; 9]; 2]) {
    let l = [f32::from(left[0]), f32::from(left[1]), f32::from(left[2])];
    let r = [
        f32::from(right[0]),
        f32::from(right[1]),
        f32::from(right[2]),
    ];

    anaglyph[0] = to_u8(dot3(l, m[0][0], m[0][1], m[0][2]) + dot3(r, m[1][0], m[1][1], m[1][2]));
    anaglyph[1] = to_u8(dot3(l, m[0][3], m[0][4], m[0][5]) + dot3(r, m[1][3], m[1][4], m[1][5]));
    anaglyph[2] = to_u8(dot3(l, m[0][6], m[0][7], m[0][8]) + dot3(r, m[1][6], m[1][7], m[1][8]));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_matrix(scale: f32) -> [[f32; 9]; 2] {
        [
            [scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    }

    #[test]
    fn combine_slices_rounds_instead_of_truncating() {
        // 255 * 0.5 = 127.5 must round to 128, not truncate to 127.
        let mut out = [0u8; 3];
        combine_slices(&[255, 0, 0], &[0, 0, 0], &mut out, &gray_matrix(0.5));
        assert_eq!(out[0], 128);
    }

    #[test]
    fn combine_slices_saturates_instead_of_wrapping() {
        // A coefficient sum > 1 would wrap with a bare `as u8` cast
        // (510 as u8 == 254, 255 * 2 == 510). It must clamp to 255 instead.
        let mut out = [0u8; 3];
        combine_slices(&[255, 0, 0], &[255, 0, 0], &mut out, &gray_matrix(1.0));
        assert_eq!(out[0], 255);
    }

    #[test]
    fn combine_slices_clamps_negative_sums_to_zero() {
        let mut out = [0u8; 3];
        combine_slices(&[0, 0, 0], &[0, 0, 0], &mut out, &gray_matrix(-1.0));
        assert_eq!(out[0], 0);
    }

    #[test]
    fn offset_of_min_i32_is_an_error_not_an_overflow() {
        let img = RgbImage::new(8, 8);
        let err = left_right_to_anaglyph_offset(
            &img,
            &img,
            AnaglyphType::Color,
            Offset { x: i32::MIN, y: 0 },
        )
        .unwrap_err();
        assert!(err.contains("X offset"), "unexpected error: {err}");
    }

    #[test]
    fn size_mismatch_is_an_error() {
        let a = RgbImage::new(8, 8);
        let b = RgbImage::new(7, 8);
        assert!(left_right_to_anaglyph(&a, &b, AnaglyphType::Color).is_err());
    }

    #[test]
    fn offset_crop_keeps_the_overlap() {
        // Left white, right black; color matrix keeps left red only. With a
        // +2 x offset the right image is sampled 2 px to the right, so the
        // output is 6 px wide and pure red.
        let left = RgbImage::from_pixel(8, 4, image::Rgb([255, 255, 255]));
        let right = RgbImage::from_pixel(8, 4, image::Rgb([0, 0, 0]));
        let out = left_right_to_anaglyph_offset(
            &left,
            &right,
            AnaglyphType::Color,
            Offset { x: 2, y: 0 },
        )
        .unwrap();
        assert_eq!(out.dimensions(), (6, 4));
        for px in out.pixels() {
            assert_eq!(px.0, [255, 0, 0]);
        }
    }

    #[test]
    fn from_str_round_trips() {
        assert_eq!(
            "half-color".parse::<AnaglyphType>().unwrap(),
            AnaglyphType::HalfColor
        );
        assert_eq!(
            AnaglyphType::HalfColor
                .name()
                .parse::<AnaglyphType>()
                .unwrap(),
            AnaglyphType::HalfColor
        );
        assert_eq!(
            "counter-clockwise".parse::<VideoDirection>().unwrap(),
            VideoDirection::CounterClockwise
        );
        for choice in [BlenderChoice::Auto, BlenderChoice::Gpu, BlenderChoice::Cpu] {
            assert_eq!(choice.name().parse::<BlenderChoice>().unwrap(), choice);
        }
        assert_eq!(BlenderChoice::default(), BlenderChoice::Auto);
        assert!("sepia".parse::<AnaglyphType>().is_err());
        assert!("sideways".parse::<VideoDirection>().is_err());
        assert!("turbo".parse::<BlenderChoice>().is_err());
    }
}
