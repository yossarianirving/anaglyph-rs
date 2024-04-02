use image::{self, Pixel, RgbImage};

#[derive(Debug)]
pub enum AnaglyphType {
    True,
    GrayScale,
    Optimized,
    Color,
    HalfColor,
}

const TRUE_MATRIX: [[f32; 9]; 2] = [ [ 0.299, 0.587, 0.114, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.299, 0.587, 0.114 ] ];
const GRAY_SCALE_MATRIX: [[f32; 9]; 2] = [ [ 0.299, 0.587, 0.114, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114 ] ];
const OPTIMIZED_MATRIX: [[f32; 9]; 2] = [ [ 0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ] ];
const COLOR_MATRIX: [[f32; 9]; 2] = [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ] ];
const HALF_COLOR_MATRIX: [[f32; 9]; 2] = [ [ 0.299, 0.587, 0.114, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ] ];



pub fn left_right_to_anaglyph(left_image: RgbImage, right_image: RgbImage, anaglyph_type: AnaglyphType) -> RgbImage {
    if left_image.height() != right_image.height() || left_image.width() != right_image.width() {
        panic!("Left and right images must be same size")
    }

    let height = left_image.height();
    let width = left_image.width();

    // new rgbImage
    let mut anaglyph = RgbImage::new(width, height);

    for x in 0..width {
        for y in 0..height {
            let left_slice = left_image.get_pixel(x, y).channels();
            let right_slice = right_image.get_pixel(x, y).channels();
            let anaglyph_slice = anaglyph.get_pixel_mut(x, y).channels_mut();
            combine_slices(left_slice, right_slice, anaglyph_slice, &anaglyph_type)
        }
    }

    anaglyph
}


fn combine_slices(left: &[u8], right: &[u8], anaglyph: &mut [u8], anaglyph_type: &AnaglyphType) {
    let m: &[[f32; 9]; 2] = match anaglyph_type {
        AnaglyphType::True => &TRUE_MATRIX,
        AnaglyphType::GrayScale => &GRAY_SCALE_MATRIX,
        AnaglyphType::Optimized => &OPTIMIZED_MATRIX,
        AnaglyphType::Color => &COLOR_MATRIX,
        AnaglyphType::HalfColor => &HALF_COLOR_MATRIX
    };

    let l: [f32; 3] = [left[0] as f32, left[1] as f32, left[2] as f32];
    let r: [f32; 3] = [right[0] as f32, right[1] as f32, right[2] as f32];
    // red channel
    anaglyph[0] = (l[0]*m[0][0] + l[1]*m[0][1] + l[2]*m[0][2] + r[0]*m[1][0] + r[1]*m[1][1] + r[2]*m[1][2]) as u8;
    // blue channel
    anaglyph[1] = (l[0]*m[0][3] + l[1]*m[0][4] + l[2]*m[0][5] + r[0]*m[1][3] + r[1]*m[1][4] + r[2]*m[1][5]) as u8;
    // green channel
    anaglyph[2] = (l[0]*m[0][6] + l[1]*m[0][7] + l[2]*m[0][8] + r[0]*m[1][6] + r[0]*m[1][7] + r[2]*m[1][8]) as u8;
}