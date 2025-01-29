use image::{self, Pixel, RgbImage};

#[derive(Debug, Clone, Copy)]
pub enum AnaglyphType {
    True,
    GrayScale,
    Optimized,
    Color,
    HalfColor,
}

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

pub enum VideoDirection {
    Clockwise,
    CounterClockwise,
}

pub fn anaglyph_type_to_matrix(anaglyph_type: AnaglyphType) -> &'static [[f32; 9]; 2] {
    match anaglyph_type {
        AnaglyphType::True => &TRUE_MATRIX,
        AnaglyphType::GrayScale => &GRAY_SCALE_MATRIX,
        AnaglyphType::Optimized => &OPTIMIZED_MATRIX,
        AnaglyphType::Color => &COLOR_MATRIX,
        AnaglyphType::HalfColor => &HALF_COLOR_MATRIX,
    }
}

pub fn left_right_to_anaglyph_offset(
    left_image: &RgbImage,
    right_image: &RgbImage,
    anaglyph_type: AnaglyphType,
    offset: Offset,
) -> RgbImage {
    if left_image.height() != right_image.height() || left_image.width() != right_image.width() {
        panic!("Left and right images must be same size")
    }
    let new_height = match offset.y.abs() as u32 {
        i if i >= left_image.height() => panic!("Y offset cannot be greater than height"),
        i => left_image.height() - i,
    };
    let new_width = match offset.x.abs() as u32 {
        i if i >= left_image.width() => panic!("X offset cannot be greater than width"),
        i => left_image.width() - i,
    };

    let mut anaglyph = RgbImage::new(new_width, new_height);

    let left_x_offset = match offset.x {
        i if i >= 0 => i as u32,
        _ => 0,
    };
    let left_y_offset = match offset.y {
        i if i >= 0 => i as u32,
        _ => 0,
    };
    let right_x_offset = match offset.x {
        i if i < 0 => i.abs() as u32,
        _ => 0,
    };
    let right_y_offset = match offset.y {
        i if i < 0 => i.abs() as u32,
        _ => 0,
    };

    let anaglyph_matrix = anaglyph_type_to_matrix(anaglyph_type);

    for x in 0..new_width {
        for y in 0..new_height {
            let left_slice = left_image
                .get_pixel(x + left_x_offset, y + left_y_offset)
                .channels();
            let right_slice = right_image
                .get_pixel(x + right_x_offset, y + right_y_offset)
                .channels();
            let anaglyph_slice = anaglyph.get_pixel_mut(x, y).channels_mut();
            combine_slices(left_slice, right_slice, anaglyph_slice, anaglyph_matrix)
        }
    }

    anaglyph
}

pub fn left_right_to_anaglyph(
    left_image: &RgbImage,
    right_image: &RgbImage,
    anaglyph_type: AnaglyphType,
) -> RgbImage {
    if left_image.height() != right_image.height() || left_image.width() != right_image.width() {
        panic!("Left and right images must be same size")
    }

    let height = left_image.height();
    let width = left_image.width();

    // new rgbImage
    let mut anaglyph = RgbImage::new(width, height);

    let anaglyph_matrix = anaglyph_type_to_matrix(anaglyph_type);

    for x in 0..width {
        for y in 0..height {
            let left_slice = left_image.get_pixel(x, y).channels();
            let right_slice = right_image.get_pixel(x, y).channels();
            let anaglyph_slice = anaglyph.get_pixel_mut(x, y).channels_mut();
            combine_slices(left_slice, right_slice, anaglyph_slice, anaglyph_matrix)
        }
    }

    anaglyph
}

pub fn combine_slices(
    left: &[u8],
    right: &[u8],
    anaglyph: &mut [u8],
    anaglyph_matrix: &[[f32; 9]; 2],
) {
    let l: [f32; 3] = [left[0] as f32, left[1] as f32, left[2] as f32];
    let r: [f32; 3] = [right[0] as f32, right[1] as f32, right[2] as f32];

    anaglyph[0] = (l[0] * anaglyph_matrix[0][0]
        + l[1] * anaglyph_matrix[0][1]
        + l[2] * anaglyph_matrix[0][2]
        + r[0] * anaglyph_matrix[1][0]
        + r[1] * anaglyph_matrix[1][1]
        + r[2] * anaglyph_matrix[1][2]) as u8;

    anaglyph[1] = (l[0] * anaglyph_matrix[0][3]
        + l[1] * anaglyph_matrix[0][4]
        + l[2] * anaglyph_matrix[0][5]
        + r[0] * anaglyph_matrix[1][3]
        + r[1] * anaglyph_matrix[1][4]
        + r[2] * anaglyph_matrix[1][5]) as u8;

    anaglyph[2] = (l[0] * anaglyph_matrix[0][6]
        + l[1] * anaglyph_matrix[0][7]
        + l[2] * anaglyph_matrix[0][8]
        + r[0] * anaglyph_matrix[1][6]
        + r[1] * anaglyph_matrix[1][7]
        + r[2] * anaglyph_matrix[1][8]) as u8;
}
