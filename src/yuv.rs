//! YUV420 → RGBA conversion for raw decoder output.
//!
//! WebKit (`VideoFrame.copyTo` on Safari/iOS/WebKitGTK) ignores the
//! requested output format and hands back the frame's native NV12/I420
//! planes. The browser pipeline feeds those planes here instead of
//! rasterizing through a canvas, which measured ~7x cheaper per frame
//! (see BENCH.md "Engine notes").
//!
//! Integer-only arithmetic (10-bit fixed point, limited-range BT.601 /
//! BT.709 matrices) so results are exact and identical on every platform;
//! luma is expanded from the video range (16–235) to full scale.

/// Which Y′C′B′C′→R′G′B′ matrix to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YuvStandard {
    /// BT.601 (SD; also `smpte170m` / `bt470bg`).
    Bt601,
    /// BT.709 (HD; the assumed default for unknown-tagged HD content).
    Bt709,
}

/// Chroma siting of the source planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawYuvFormat {
    /// Semi-planar: full luma plane, then one interleaved UV plane.
    Nv12,
    /// Planar: Y, then U, then V.
    I420,
}

/// Geometry of a raw 4:2:0 frame inside one byte buffer.
///
/// `x`/`y` locate the top-left corner of the region to convert, in coded
/// frame pixels; `width`/`height` are the output dimensions. Output
/// column `ox` samples chroma column `(x + ox) / 2`, so an odd `x` reads
/// one chroma column past `x + width / 2` and bounds checks account for
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawYuvLayout {
    pub format: RawYuvFormat,
    /// Byte offset of the luma plane.
    pub y_offset: usize,
    /// Bytes between the start of successive luma rows.
    pub y_stride: usize,
    /// Byte offset of the chroma plane (UV pairs for NV12, U for I420).
    pub uv_offset: usize,
    /// Bytes between the start of successive chroma rows.
    pub uv_stride: usize,
    /// For I420: byte offset of the V plane (unused for NV12).
    pub v_offset: usize,
    /// Top-left corner of the region to convert, in source pixels.
    pub x: usize,
    pub y: usize,
    /// Output width/height in pixels.
    pub width: usize,
    pub height: usize,
}

/// 10-bit fixed-point coefficients: `(yb, vr, vg, ug, ub)`.
fn coefficients(standard: YuvStandard) -> (i32, i32, i32, i32, i32) {
    match standard {
        // R = 1.164(Y-16) + 1.596(V-128); G = 1.164(Y-16) - 0.391(U-128)
        //    - 0.813(V-128); B = 1.164(Y-16) + 2.018(U-128)   (BT.601)
        YuvStandard::Bt601 => (1192, 1634, -833, -401, 2066),
        // R = 1.164(Y-16) + 1.793(V-128); G = ... - 0.213(U-128)
        //    - 0.533(V-128); B = ... + 2.112(U-128)            (BT.709)
        YuvStandard::Bt709 => (1192, 1836, -546, -218, 2163),
    }
}

/// Convert a 4:2:0 region to dense RGBA (`dst.len() == width * height * 4`).
///
/// Returns `false` (writing nothing) when the layout does not fit the
/// provided planes or destination, so callers can validate and fall back.
pub fn yuv420_to_rgba(
    dst: &mut [u8],
    planes: &[u8],
    layout: &RawYuvLayout,
    standard: YuvStandard,
) -> bool {
    if layout.width == 0 || layout.height == 0 || dst.len() != layout.width * layout.height * 4 {
        return false;
    }
    // Chroma columns covered by the crop: luma column `sx` samples chroma
    // column `sx >> 1`, so an odd `x` needs one extra column at the right.
    let parity = layout.x & 1;
    let first_c = layout.x / 2;
    let span_c = (layout.x + layout.width - 1) / 2 - first_c + 1;
    // Last chroma row touched by rows `y..y + height`.
    let last_c_row = (layout.y + layout.height - 1) / 2;
    // Last byte (exclusive) of each plane the crop reads, relative to the
    // plane's own offset within `planes`.
    let needed_y = (layout.y + layout.height - 1) * layout.y_stride + layout.x + layout.width;
    let last_c = last_c_row * layout.uv_stride;
    let (needed_c, needed_v) = match layout.format {
        RawYuvFormat::Nv12 => (last_c + first_c * 2 + span_c * 2, 0),
        RawYuvFormat::I420 => (last_c + first_c + span_c, last_c + first_c + span_c),
    };
    let fits = layout.y_offset + needed_y <= planes.len()
        && layout.uv_offset + needed_c <= planes.len()
        && match layout.format {
            RawYuvFormat::Nv12 => true,
            RawYuvFormat::I420 => layout.v_offset + needed_v <= planes.len(),
        };
    if !fits {
        return false;
    }

    let (yb, vr, vg, ug, ub) = coefficients(standard);
    let y_plane = &planes[layout.y_offset..];
    let chroma = &planes[layout.uv_offset..];
    let v_plane = &planes[layout.v_offset..];

    for oy in 0..layout.height {
        let sy = layout.y + oy;
        let yrow = &y_plane[sy * layout.y_stride + layout.x..][..layout.width];
        // Two output rows share one chroma row (4:2:0).
        let crow = (sy / 2) * layout.uv_stride;
        let (urow, vrow, cstep): (&[u8], &[u8], usize) = match layout.format {
            RawYuvFormat::Nv12 => {
                // Interleaved UV pairs; read as one slice, u at even bytes,
                // v at odd bytes (hence `vrow` offset by one, same `cstep`).
                let pair = &chroma[crow + first_c * 2..][..span_c * 2];
                (pair, &pair[1..], 2)
            }
            RawYuvFormat::I420 => (
                &chroma[crow + first_c..][..span_c],
                &v_plane[crow + first_c..][..span_c],
                1,
            ),
        };
        let out = &mut dst[oy * layout.width * 4..][..layout.width * 4];
        for ox in 0..layout.width {
            // 10-bit fixed point, rounded; values then clamped to 0..=255.
            let y = ((i32::from(yrow[ox]) - 16) * yb + 512) >> 10;
            let cx = ((ox + parity) / 2) * cstep;
            let u = i32::from(urow[cx]) - 128;
            let v = i32::from(vrow[cx]) - 128;
            let r = y + ((vr * v + 512) >> 10);
            let g = y + ((vg * v + ug * u + 512) >> 10);
            let b = y + ((ub * u + 512) >> 10);
            let o = ox * 4;
            out[o] = clamp16(r);
            out[o + 1] = clamp16(g);
            out[o + 2] = clamp16(b);
            out[o + 3] = 255;
        }
    }
    true
}

/// Exact clamp of a value known to be within i16 to 0..=255.
#[inline]
fn clamp16(v: i32) -> u8 {
    if v < 0 {
        0
    } else if v > 255 {
        255
    } else {
        v as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_nv12(w: usize, h: usize) -> RawYuvLayout {
        RawYuvLayout {
            format: RawYuvFormat::Nv12,
            y_offset: 0,
            y_stride: w,
            uv_offset: w * h,
            uv_stride: w,
            v_offset: 0,
            x: 0,
            y: 0,
            width: w,
            height: h,
        }
    }

    #[test]
    fn neutral_gray_roundtrip() {
        // Y=128 mid-grey, neutral chroma -> rgb ~ (128..130) for both matrices.
        let (w, h) = (4, 2);
        let mut planes = vec![0u8; w * h + w * h / 2];
        for p in planes[..w * h].iter_mut() {
            *p = 128;
        }
        for p in planes[w * h..].iter_mut() {
            *p = 128;
        }
        let mut dst = vec![0u8; w * h * 4];
        assert!(yuv420_to_rgba(
            &mut dst,
            &planes,
            &layout_nv12(w, h),
            YuvStandard::Bt709
        ));
        for px in dst.chunks_exact(4) {
            assert!((127..=131).contains(&px[0]), "{:?}", px);
            assert_eq!(px[0], px[1]);
            assert_eq!(px[1], px[2]);
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn saturated_primaries() {
        // Strongly red chroma under BT.709 that stays inside gamut enough
        // to check all three channels: Y=150, U=110, V=170 -> ~(231, 137, 118).
        let (w, h) = (2, 2);
        let mut planes = vec![0u8; w * h + w * h / 2];
        for p in planes[..w * h].iter_mut() {
            *p = 150;
        }
        for chunk in planes[w * h..].chunks_exact_mut(2) {
            chunk[0] = 110;
            chunk[1] = 170;
        }
        let mut dst = vec![0u8; w * h * 4];
        assert!(yuv420_to_rgba(
            &mut dst,
            &planes,
            &layout_nv12(w, h),
            YuvStandard::Bt709
        ));
        for px in dst.chunks_exact(4) {
            assert!((229..=233).contains(&px[0]), "r {:?}", px);
            assert!((135..=139).contains(&px[1]), "g {:?}", px);
            assert!((116..=120).contains(&px[2]), "b {:?}", px);
            assert_eq!(px[3], 255);
        }
        // Fully out-of-gamut red clamps instead of wrapping.
        for p in planes[..w * h].iter_mut() {
            *p = 235;
        }
        for chunk in planes[w * h..].chunks_exact_mut(2) {
            chunk[0] = 16;
            chunk[1] = 240;
        }
        assert!(yuv420_to_rgba(
            &mut dst,
            &planes,
            &layout_nv12(w, h),
            YuvStandard::Bt709
        ));
        assert_eq!(dst[0], 255);
    }

    #[test]
    fn i420_matches_nv12_for_same_planes() {
        let (w, h) = (6, 4);
        let mut nv12 = vec![0u8; w * h * 3 / 2];
        for (i, p) in nv12.iter_mut().enumerate() {
            *p = (i * 37 + 11) as u8 & 0xf0; // keep chroma calm-ish
        }
        // Deinterleave into planar.
        let mut i420 = vec![0u8; w * h * 3 / 2];
        i420[..w * h].copy_from_slice(&nv12[..w * h]);
        let c = w * h;
        for j in 0..(w / 2) * (h / 2) {
            i420[c + j] = nv12[c + j * 2];
            i420[c + (w * h / 4) + j] = nv12[c + j * 2 + 1];
        }
        let mut a = vec![0u8; w * h * 4];
        let mut b = vec![0u8; w * h * 4];
        let mut lay = layout_nv12(w, h);
        assert!(yuv420_to_rgba(&mut a, &nv12, &lay, YuvStandard::Bt601));
        lay.format = RawYuvFormat::I420;
        lay.uv_stride = w / 2; // planar chroma rows are half-width
        lay.v_offset = c + w * h / 4;
        assert!(yuv420_to_rgba(&mut b, &i420, &lay, YuvStandard::Bt601));
        assert_eq!(a, b);
    }

    #[test]
    fn subrect_offset_matches_full_frame_slice() {
        let (w, h) = (8, 8);
        let mut planes = vec![0u8; w * h * 3 / 2];
        for (i, p) in planes.iter_mut().enumerate() {
            *p = (i * 53) as u8;
        }
        // Convert the 4x4 region starting at (2, 2) and compare against the
        // same region from a full-frame conversion.
        let mut full = vec![0u8; w * h * 4];
        assert!(yuv420_to_rgba(
            &mut full,
            &planes,
            &layout_nv12(w, h),
            YuvStandard::Bt709
        ));
        let mut sub = vec![0u8; 4 * 4 * 4];
        let lay = RawYuvLayout {
            x: 2,
            y: 2,
            width: 4,
            height: 4,
            ..layout_nv12(w, h)
        };
        assert!(yuv420_to_rgba(&mut sub, &planes, &lay, YuvStandard::Bt709));
        for oy in 0..4 {
            let src = &full[((oy + 2) * w + 2) * 4..((oy + 2) * w + 6) * 4];
            assert_eq!(&sub[oy * 16..oy * 16 + 16], src);
        }
    }

    #[test]
    fn odd_crop_from_even_coded_frame_matches_full_conversion() {
        // Real coded frames are even-sized (NV12 cannot code odd widths), so
        // the odd-dims visible-rect case is a crop like coded 482x272 with
        // visible 480x270 at (1, 1): odd x must map chroma by source column,
        // matching what a full-frame conversion produces at those pixels.
        let (cw, chh) = (482, 272);
        let mut planes = vec![0u8; cw * chh + cw * (chh / 2)];
        for (i, p) in planes[..cw * chh].iter_mut().enumerate() {
            *p = (i % 251) as u8;
        }
        for (i, p) in planes[cw * chh..].iter_mut().enumerate() {
            *p = 128 + (i % 20) as u8; // alternating u/v, gentle range
        }
        let full_layout = layout_nv12(cw, chh);
        let mut full = vec![0u8; cw * chh * 4];
        assert!(yuv420_to_rgba(
            &mut full,
            &planes,
            &full_layout,
            YuvStandard::Bt709
        ));

        let mut dst = vec![0u8; 480 * 270 * 4];
        let mut lay = layout_nv12(480, 270);
        lay.y_stride = cw;
        lay.uv_offset = cw * chh;
        lay.uv_stride = cw;
        lay.x = 1;
        lay.y = 1;
        assert!(yuv420_to_rgba(&mut dst, &planes, &lay, YuvStandard::Bt709));
        for oy in 0..270 {
            let src_row = (oy + 1) * cw + 1;
            assert_eq!(
                &dst[oy * 480 * 4..][..480 * 4],
                &full[src_row * 4..][..480 * 4],
                "row {oy}"
            );
        }
        assert!(dst.iter().skip(3).step_by(4).all(|&a| a == 255));
    }

    #[test]
    fn rejects_short_planes() {
        let (w, h) = (4, 2);
        let planes = vec![0u8; w * h + w * h / 2 - 1];
        let mut dst = vec![0u8; w * h * 4];
        assert!(!yuv420_to_rgba(
            &mut dst,
            &planes,
            &layout_nv12(w, h),
            YuvStandard::Bt709
        ));
    }
}
