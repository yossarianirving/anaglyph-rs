# Anaglyph Converter

This program is a Rust-based utility for converting images and certain videos into Red/Cyan anaglyphs. An anaglyph is a type of 3-D image where the left and right images are superimposed in (usually) complementary colors and viewed with corresponding filters on each eye. The most commonly used (and straight forward) method is to take the red channel of the left image and add that to the blue/green channels of the right image. A viewer can then view the 3-D image through a pair of red/cyan glasses. 

## Building

If you are not using the video feature, building `anaglyph-rs` is straightforward with cargo.

```sh
cargo build --release
```

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

If the video feature in enabled, a standard video can be converted to a pseudo 3-D video with some **strong** caveats. First, the camera must be moving in an arc in a consistent direction throughout the entire video. Second, the subjects and background should be mostly static. An ideal video for this program is one taken by a 360 photo booth, though those videos often change directions for a "boomerang" effect so some editing is necessary.

```sh
cargo run --features video --release -- --video test/short.mp4 --out hs.mp4
```

Two flags control the hardware pipeline for video: `-e/--encoder auto|hardware|software` selects the output encoder (`auto`, the default, prefers VAAPI and falls back to libx264), and `-D/--decode cpu|hardware` selects the decoder (CPU is the default; see the benchmark note above).