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
//!   const out = await convert_gif_to_anaglyph(new Uint8Array(bytes), "clockwise", "color", "cpu");
//!   const url = URL.createObjectURL(new Blob([out], { type: "image/gif" }));
//! </script>
//! ```

use wasm_bindgen::prelude::*;

use crate::anaglyph::{AnaglyphType, BlenderChoice, VideoDirection};
use crate::gif::convert_gif_buffer_to_anaglyph_async;
use std::str::FromStr;

/// Convert an animated GIF (given as raw bytes) into a red/cyan anaglyph GIF,
/// returned as a `Uint8Array`.
///
/// `direction` is `"clockwise"` or `"counter-clockwise"`; `anaglyph_type` is
/// one of `"color"`, `"half-color"`, `"grayscale"`, `"optimized"`, `"true"`.
///
/// `blend` picks the blending backend: `"cpu"` (the default when omitted)
/// runs a single-threaded CPU loop, `"gpu"` requires WebGPU and fails when
/// the browser does not provide it, `"auto"` uses WebGPU where available and
/// falls back to the CPU otherwise.
///
/// The returned promise resolves once the whole gif has been converted;
/// nothing blocks the main thread meanwhile.
#[wasm_bindgen]
pub async fn convert_gif_to_anaglyph(
    gif: &[u8],
    direction: &str,
    anaglyph_type: &str,
    blend: Option<String>,
) -> Result<Vec<u8>, JsValue> {
    let direction = VideoDirection::from_str(direction).map_err(|e| JsValue::from_str(&e))?;
    let anaglyph_type = AnaglyphType::from_str(anaglyph_type).map_err(|e| JsValue::from_str(&e))?;
    // CPU is the default: WebGPU setup has a one-time startup cost and the
    // caller has to opt in to it explicitly.
    let blend =
        BlenderChoice::from_str(blend.as_deref().unwrap_or("cpu")).map_err(|e| JsValue::from_str(&e))?;

    convert_gif_buffer_to_anaglyph_async(gif, direction, anaglyph_type, blend)
        .await
        .map_err(|e| JsValue::from_str(&e))
}
