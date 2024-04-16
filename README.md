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