//! Browser entry points exposed through wasm-bindgen.
//!
//! Build with `wasm-pack build --target web --features wasm` and call from
//! JavaScript:
//!
//! ```html
//! <script type="module">
//!   import init, { convert_gif_to_anaglyph } from "./pkg/anaglyph_rs.js";
//!   await init();
//!   const bytes = await (await fetch("input.gif")).arrayBuffer();
//!   const out = await convert_gif_to_anaglyph(new Uint8Array(bytes), "clockwise", "color");
//!   const url = URL.createObjectURL(new Blob([out], { type: "image/gif" }));
//! </script>
//! ```

use wasm_bindgen::prelude::*;

use crate::anaglyph::{AnaglyphType, VideoDirection};
use crate::gif::convert_gif_buffer_to_anaglyph_async;

/// Convert an animated GIF (given as raw bytes) into a red/cyan anaglyph GIF,
/// returned as a `Uint8Array`.
///
/// `direction` is `"clockwise"` or `"counter-clockwise"`; `anaglyph_type` is
/// one of `"color"`, `"half-color"`, `"grayscale"`, `"optimized"`, `"true"`.
///
/// Blending runs on WebGPU when the browser provides it and falls back to a
/// single-threaded CPU loop otherwise. The returned promise resolves once the
/// whole gif has been converted; nothing blocks the main thread meanwhile.
#[wasm_bindgen]
pub async fn convert_gif_to_anaglyph(
    gif: &[u8],
    direction: &str,
    anaglyph_type: &str,
) -> Result<Vec<u8>, JsValue> {
    let direction = match direction {
        "clockwise" => VideoDirection::Clockwise,
        "counter-clockwise" => VideoDirection::CounterClockwise,
        other => return Err(JsValue::from_str(&format!("unknown direction: {other}"))),
    };
    let anaglyph_type = match anaglyph_type {
        "color" => AnaglyphType::Color,
        "half-color" => AnaglyphType::HalfColor,
        "grayscale" => AnaglyphType::GrayScale,
        "optimized" => AnaglyphType::Optimized,
        "true" => AnaglyphType::True,
        other => {
            return Err(JsValue::from_str(&format!(
                "unknown anaglyph type: {other}"
            )))
        }
    };

    convert_gif_buffer_to_anaglyph_async(gif, direction, anaglyph_type)
        .await
        .map_err(|e| JsValue::from_str(&e))
}
