# Anaglyph Converter

This program is a Rust-based utility for converting images and certain videos into Red/Cyan anaglyphs. An anaglyph is a type of 3-D image where the left and right images are superimposed in (usually) complementary colors and viewed with corresponding filters on each eye. The most commonly used (and straight forward) method is to take the red channel of the left image and add that to the blue/green channels of the right image. A viewer can then view the 3-D image through a pair of red/cyan glasses. 

## Building

Plain `cargo build --release` builds the image and gif converters with a
pure-CPU blend and has no system dependencies.

```sh
cargo build --release
```

Cargo features:

| Feature | What it adds |
|---|---|
| `gpu` | Blends images, gifs and video frames with a `wgpu` compute shader. No ffmpeg dependency, so it also builds for `wasm32-unknown-unknown` (browser WebGPU). |
| `parallel` | Rayon-parallel CPU blending, used as the fallback when no GPU is available. |
| `video` | Video conversion via video-rs (needs ffmpeg dev libraries) plus VAAPI hardware encoding. Implies `gpu` and `parallel`. |
| `wasm` | `wasm-bindgen` entry points for the browser demo. Implies `gpu`. |

Building with the video feature requires you to install a ffmpeg dev libraries on your machine. The `ffmpeg-next` crate is an indirect dependency and has (https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building#dependencies)[good instructions on its wiki]. After installing the necessary libraries, run the following command:

```sh
cargo build --release --features video
```

The video pipeline uses hardware acceleration where available: the anaglyph combination is computed by a `wgpu` compute shader (Vulkan/DX12/Metal), and the output is encoded with the GPU VAAPI encoder (`h264_vaapi`) by default, falling back to software (libx264) when no VAAPI device is present. Decoding runs on the CPU by default; GPU decoding is available with `-D hardware` but is often *slower* end-to-end because every frame has to be downloaded back to system memory for the blend step. If no suitable GPU is found the blend falls back to a parallel CPU implementation, so it works everywhere, just slower.

### Encoder benchmarks

Measured on an AMD Radeon 8060S iGPU (Strix Halo), half-color mode, encoder backend selected with `-e`:

| Input | Encoder | Wall | CPU time | Output |
|---|---|---|---|---|
| 1080p, 599 frames | `-e hardware` | 3.7 s | 7.2 s | 18 MiB |
| 1080p, 599 frames | `-e software` | 2.8–6.0 s | 20.5 s | 13 MiB |
| 4K, 299 frames | `-e hardware` | 7.2 s | 15.7 s | 36 MiB |
| 4K, 299 frames | `-e software` | 7.2 s | 40.3 s | 24 MiB |

Hardware encoding cuts CPU time by roughly 3x with visually identical output (≈43 dB PSNR vs the software-encoded file) at the cost of ~40% more bytes (CQP qp=23 vs x264 CRF 23). Wall time is comparable at 1080p and dominated by decode + blend at 4K; the encoder is no longer the bottleneck. VAAPI's `low_power` mode is not exposed because many Mesa drivers do not implement it.

## Usage

To convert the test left and right images into an anaglyph run the following command and wear a pair of red/cyan glasses to view.

```sh
cargo run --release  -- -l test/l.jpeg -r test/r.jpeg -o new.jpeg
```

You may notice the image doesn't look that 3-D and objects are sticking out from the screen slightly. This is because no depth adjustment has been applied. To adjust the depth, pass in the `--offset-x` parameter along with the number of pixels you want to shift the image by. In most cases, you will want a positive number but negative numbers are allowed.

```sh
cargo run  -- -l test/l.jpeg -r test/r.jpeg -o new.jpeg --offset-x 15 --offset-y -15
```

In the (current) test images, there is a slight tilt missalignment so adding a y-axis offset was necessary.

Animated gifs are converted frame-to-frame (each frame is blended with the previous one):

```sh
cargo run --release -- -g test/3drot.gif -o out.gif
```

With the `gpu` feature the per-frame blend runs on the GPU, but note that on a
desktop CPU the gif encoder dominates the runtime (~90% of it), so GPU vs CPU
blending is roughly a wash natively. The GPU path matters most in the browser,
where the CPU fallback is single-threaded (see below).

If the video feature in enabled, a standard video can be converted to a pseudo 3-D video with some **strong** caveats. First, the camera must be moving in an arc in a consistent direction throughout the entire video. Second, the subjects and background should be mostly static. An ideal video for this program is one taken by a 360 photo booth, though those videos often change directions for a "boomerang" effect so some editing is necessary.

```sh
cargo run --features video --release -- --video test/short.mp4 --out hs.mp4
```

Two flags control the hardware pipeline for video: `-e/--encoder auto|hardware|software` selects the output encoder (`auto`, the default, prefers VAAPI and falls back to libx264), and `-D/--decode cpu|hardware` selects the decoder (CPU is the default; see the benchmark note above).

## Web (wasm + WebGPU)

The gif converter also runs in the browser: the same `wgpu` compute shader
blends frames through WebGPU, compiled to wasm with `wasm-bindgen`. Without
WebGPU the code falls back to a single-threaded CPU loop (a warning is logged
to the JS console).

Build the package (a `wasm-bindgen-cli` version matching the `wasm-bindgen`
crate is required; `wasm-pack` 0.15 currently passes flags that recent cargo
versions reject, so drive the tools directly):

```sh
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --release --features wasm --lib
wasm-bindgen target/wasm32-unknown-unknown/release/anaglyph_rs.wasm --out-dir pkg --target web
```

Then serve the repository root over `http://localhost` (WebGPU requires a
secure context; `localhost` counts) and open `web/index.html`. It auto-converts
`web/3drot.gif` (copy any gif there, e.g. `cp test/3drot.gif web/`) and also
has a file picker.

From JavaScript, the exported async function takes the gif bytes and option
names and resolves with the converted gif bytes:

```js
import init, { convert_gif_to_anaglyph } from "./pkg/anaglyph_rs.js";
await init();
const bytes = new Uint8Array(await (await fetch("input.gif")).arrayBuffer());
const out = await convert_gif_to_anaglyph(bytes, "clockwise", "color");
```