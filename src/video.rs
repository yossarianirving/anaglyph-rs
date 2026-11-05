use ndarray::Array3;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::mpsc::{self, SyncSender};
use std::thread::{self, JoinHandle};

use video_rs::decode::DecoderBuilder;
use video_rs::encode::{Encoder, Settings};
use video_rs::ffmpeg::Rational;
use video_rs::hwaccel::HardwareAccelerationDeviceType;
use video_rs::Time;

use crate::anaglyph::{
    anaglyph_type_to_matrix, combine_slices, AnaglyphType, BlenderChoice, VideoDirection,
};
use crate::gpu::GpuBlender;
use crate::hwencode::{self, EncodeOptions, HwCodec, HwEncoder};
use crate::motion::{DirectionTracker, MotionEstimator, MotionSample, Trend};

/// Device types to try for hardware-accelerated decoding, in order.
const HWACCEL_PREFERENCE: [HardwareAccelerationDeviceType; 3] = [
    HardwareAccelerationDeviceType::VaApi,
    HardwareAccelerationDeviceType::Cuda,
    HardwareAccelerationDeviceType::Vdpau,
];

/// Depth of the queue between the blending thread and the encoding thread.
/// Caps memory in flight at roughly `queue * frame size` and provides
/// backpressure when the encoder falls behind.
const ENCODE_QUEUE_DEPTH: usize = 8;

/// Depth of the queue between the decode thread and the blending loop. Also
/// bounded: decoding (especially on the GPU) outruns blending by far, and an
/// unbounded queue would pile up raw frames until memory runs out.
const DECODE_QUEUE_DEPTH: usize = 8;

/// Which encoder backend to use for the output video.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EncoderChoice {
    /// Prefer a hardware encoder (currently VAAPI), fall back to software.
    Auto,
    /// Require a hardware encoder; fail if none is available.
    Hardware,
    /// Always use the software (libx264) encoder.
    Software,
}

/// Hardware encoders tried by [`EncoderChoice::Auto`] and
/// [`EncoderChoice::Hardware`], in order of preference. VAAPI is currently
/// the only backend; when more are added to [`hwencode`], this list grows
/// (and may diverge per target OS — native hardware first).
const HARDWARE_ORDER: [HwCodec; 1] = [HwCodec::Vaapi];

/// The active encoding backend.
enum Backend {
    Software(Box<Encoder>),
    Hardware(Box<dyn HwEncoder>),
}

impl Backend {
    /// Encode one dense RGB24 frame, returning the buffer on success so the
    /// caller keeps ownership (the encode thread recycles it as the guard
    /// frame without copying).
    fn encode(
        &mut self,
        rgb: Vec<u8>,
        time: Time,
        shape: (usize, usize),
    ) -> Result<Vec<u8>, String> {
        match self {
            Backend::Software(encoder) => {
                let (height, width) = shape;
                let frame =
                    Array3::from_shape_vec((height, width, 3), rgb).map_err(|e| e.to_string())?;
                encoder.encode(&frame, time).map_err(|e| e.to_string())?;
                // Frames built by `from_shape_vec` always have a standard
                // layout, so the raw vec offset is None and the buffer is
                // returned as-is.
                Ok(frame.into_raw_vec_and_offset().0)
            }
            Backend::Hardware(encoder) => {
                let pts = time.into_value().unwrap_or(0);
                encoder.encode(&rgb, pts)?;
                Ok(rgb)
            }
        }
    }

    fn finish(&mut self) -> Result<(), String> {
        match self {
            Backend::Software(encoder) => encoder.finish().map_err(|e| e.to_string()),
            Backend::Hardware(encoder) => encoder.finish(),
        }
    }
}

/// Summary of a finished conversion, for the caller to report.
#[derive(Debug, Clone, PartialEq)]
pub struct ConversionReport {
    /// Number of blended frames encoded.
    pub frames: u64,
    /// Decoder used, e.g. `"cpu"` or `"VaApi hardware"`.
    pub decoder: String,
    /// Blender used, e.g. `"GPU (wgpu)"` or a CPU fallback with the reason.
    pub blender: String,
    /// Encoder used, e.g. `"VAAPI (h264_vaapi)"` or a fallback with the reason.
    pub encoder: String,
    /// Detected direction reversals, empty unless auto direction was used.
    pub reversals: Vec<DirectionReversal>,
    /// Per-frame motion trace, empty unless auto direction was used.
    pub motion: Vec<MotionPoint>,
}

/// A point where a booth video changed orbit direction and the eye pairing
/// was swapped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionReversal {
    /// Index of the first output pair's newer input frame with the new
    /// direction.
    pub frame: usize,
    /// Source timestamp of the turning point in seconds.
    pub time: f64,
    pub from: VideoDirection,
    pub to: VideoDirection,
}

/// One frame's measured image motion, exported for tuning and debugging.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionPoint {
    /// Input frame index (0-based, first frame has no sample).
    pub frame: usize,
    /// Source timestamp in seconds.
    pub time: f64,
    /// Measured horizontal drift in pixels (+ = content moved right).
    pub dx: f32,
    /// False when the frame could not be matched (scene cut, textureless).
    pub reliable: bool,
    /// Detected trend after this sample.
    pub trend: Option<Trend>,
}

/// A unit of encoding work handed to the encode thread.
enum Job {
    /// Encode this RGB24 frame at the given timestamp.
    Frame(Vec<u8>, Time),
    /// Encode the previous frame again at the given (later) timestamp so the
    /// muxer can assign a duration to the final real frame (see call site).
    Guard(Time),
}

/// Which decoder mode to use for the input video.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecoderChoice {
    /// Decode on the CPU with FFmpeg's software decoder.
    Cpu,
    /// Decode with the GPU (VAAPI/CUDA/VDPAU) when available.
    Hardware,
}

/// How to pair frames into stereo views over the course of a video.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DirectionChoice {
    /// One fixed pairing for the whole video.
    Fixed(VideoDirection),
    /// Detect the camera's orbit direction from the image motion and swap the
    /// eyes at every reversal. `reverse` flips the detected pairing, for rigs
    /// whose physical rotation runs opposite to the convention below.
    Auto { reverse: bool },
}

/// Map a detected drift trend onto an eye pairing.
///
/// Convention: the booth camera rotates to keep the subject framed, and that
/// tracking rotation dominates the image motion — a background point shifts
/// by `f * Tx * (1/Zs - 1/Z)` per frame, so content drifts in the *same*
/// direction as the camera's translation (verified on real booth footage:
/// the background slides, the tracked subject stays nearly put). Image
/// content drifting right therefore means the camera moved right, the newer
/// frame is the physically *right* view, and the older frame feeds the left
/// eye (`VideoDirection::CounterClockwise`); content drifting left gives
/// `Clockwise`. `reverse` inverts the mapping. A missing trend (stationary
/// camera, or fewer than two frames) falls back to the un-reversed base
/// pairing.
fn trend_to_direction(trend: Option<Trend>, reverse: bool) -> VideoDirection {
    let clockwise_forwards = match trend {
        Some(Trend::Increasing) => false,
        Some(Trend::Decreasing) | None => true,
    };
    if clockwise_forwards != reverse {
        VideoDirection::Clockwise
    } else {
        VideoDirection::CounterClockwise
    }
}

/// A decoded frame waiting to be paired.
struct PendingFrame {
    index: usize,
    time: Time,
    rgb: Array3<u8>,
}

/// Frame pairing state shared by the fixed and auto paths: frames arrive in
/// order via `push`, completed pairs leave through the front via `peek_front`
/// and `pop_front`. Fixed mode streams with a two-frame window; auto mode
/// holds frames back until the [`DirectionTracker`] has confirmed their
/// trend (which bounds the hold to the hysteresis plus `pending_cap`).
enum Pairer {
    Fixed {
        direction: VideoDirection,
        next_index: usize,
        window: VecDeque<PendingFrame>,
    },
    Auto {
        reverse: bool,
        tracker: DirectionTracker,
        next_index: usize,
        window: VecDeque<PendingFrame>,
        motion: Vec<MotionPoint>,
    },
}

impl Pairer {
    fn fixed(direction: VideoDirection) -> Self {
        Pairer::Fixed {
            direction,
            next_index: 0,
            window: VecDeque::new(),
        }
    }

    fn auto(reverse: bool) -> Self {
        Pairer::Auto {
            reverse,
            tracker: DirectionTracker::new(Default::default()),
            next_index: 0,
            window: VecDeque::new(),
            motion: Vec::new(),
        }
    }

    /// Register the next decoded frame. `sample` is the motion against the
    /// previous frame; `None` frames (the first, or fixed mode) carry none.
    fn push(&mut self, time: Time, rgb: Array3<u8>, sample: Option<MotionSample>) {
        match self {
            Self::Fixed {
                next_index, window, ..
            } => {
                if window.len() == 2 {
                    window.pop_front();
                }
                window.push_back(PendingFrame {
                    index: *next_index,
                    time,
                    rgb,
                });
                *next_index += 1;
            }
            Self::Auto {
                tracker,
                next_index,
                window,
                motion,
                ..
            } => {
                let index = *next_index;
                *next_index += 1;
                if let Some(sample) = sample {
                    let time_secs = time.as_secs_f64();
                    tracker.push(index, time_secs, &sample);
                    motion.push(MotionPoint {
                        frame: index,
                        time: time_secs,
                        dx: sample.dx,
                        reliable: sample.reliable,
                        trend: tracker.current_trend(),
                    });
                }
                window.push_back(PendingFrame { index, time, rgb });
            }
        }
    }

    /// Whether a completed pair is available at the front.
    fn ready(&self) -> bool {
        match self {
            Self::Fixed { window, .. } => window.len() == 2,
            Self::Auto {
                window, tracker, ..
            } => {
                window.len() >= 2
                    && tracker
                        .confirmed_through()
                        .is_some_and(|c| window[1].index <= c)
            }
        }
    }

    /// The older frame's timestamp at the front. Requires `ready()`.
    fn front_time(&self) -> Time {
        self.frames().front().expect("pairer is ready").time
    }

    /// The pairing direction for the front pair. Requires `ready()`.
    fn front_direction(&self) -> VideoDirection {
        match self {
            Self::Fixed { direction, .. } => *direction,
            Self::Auto {
                tracker, reverse, ..
            } => trend_to_direction(tracker.current_trend(), *reverse),
        }
    }

    /// The front pair's dense RGB24 pixels in left/right order for
    /// `front_direction()`. Requires `ready()`; the borrow ends with the
    /// binding, so `pop_front` can run once the pixels are consumed.
    fn front_pixels(&self) -> (&[u8], &[u8]) {
        let window = self.frames();
        let previous = &window[0];
        let current = &window[1];
        match self.front_direction() {
            VideoDirection::Clockwise => (raw_rgb(&current.rgb), raw_rgb(&previous.rgb)),
            VideoDirection::CounterClockwise => (raw_rgb(&previous.rgb), raw_rgb(&current.rgb)),
        }
    }

    fn frames(&self) -> &VecDeque<PendingFrame> {
        match self {
            Self::Fixed { window, .. } | Self::Auto { window, .. } => window,
        }
    }

    fn pop_front(&mut self) {
        self.frames_mut().pop_front();
    }

    fn frames_mut(&mut self) -> &mut VecDeque<PendingFrame> {
        match self {
            Self::Fixed { window, .. } | Self::Auto { window, .. } => window,
        }
    }

    /// End of stream: commit every held-back frame to the current trend and
    /// take the reversal list and motion trace (empty in fixed mode).
    fn finish(&mut self) -> (Vec<DirectionReversal>, Vec<MotionPoint>) {
        match self {
            Self::Fixed { .. } => (Vec::new(), Vec::new()),
            Self::Auto {
                tracker,
                reverse,
                motion,
                ..
            } => {
                tracker.finish();
                let reversals = tracker
                    .take_reversals()
                    .into_iter()
                    .map(|r| DirectionReversal {
                        frame: r.frame,
                        time: r.time,
                        from: trend_to_direction(Some(r.from), *reverse),
                        to: trend_to_direction(Some(r.to), *reverse),
                    })
                    .collect();
                (reversals, std::mem::take(motion))
            }
        }
    }
}

/// Convert a video into an anaglyph video, cleaning up after a failed run.
///
/// A failure after the encoder opened the output file would otherwise leave
/// a truncated, unplayable file (or an empty stub) behind; when the file was
/// created by this call, it is removed. A pre-existing file is left alone.
pub fn convert_video_to_anaglyph(
    video: &str,
    video_out: &str,
    direction: DirectionChoice,
    anaglyph_type: AnaglyphType,
    decoder_choice: DecoderChoice,
    encoder_choice: EncoderChoice,
    blend: BlenderChoice,
) -> Result<ConversionReport, String> {
    let out = Path::new(video_out);
    let pre_existing = out.exists();
    let result = convert_video(
        video,
        video_out,
        direction,
        anaglyph_type,
        decoder_choice,
        encoder_choice,
        blend,
    );
    if result.is_err() && !pre_existing {
        let _ = std::fs::remove_file(out);
    }
    result
}

fn convert_video(
    video: &str,
    video_out: &str,
    direction: DirectionChoice,
    anaglyph_type: AnaglyphType,
    decoder_choice: DecoderChoice,
    encoder_choice: EncoderChoice,
    blend: BlenderChoice,
) -> Result<ConversionReport, String> {
    video_rs::init().map_err(|e| format!("failed to initialize ffmpeg: {e}"))?;

    let mut builder = DecoderBuilder::new(Path::new(video));
    let mut decoder_name = "cpu".to_string();
    // Note: GPU decoding downloads each frame back to system memory for the
    // blend step, which on many setups (including iGPUs) is slower than just
    // decoding on the CPU. Hardware decoding helps for high resolutions or
    // expensive codecs, so make it opt-in.
    if let DecoderChoice::Hardware = decoder_choice {
        if let Some(device) = HWACCEL_PREFERENCE
            .iter()
            .find(|device| device.is_available())
        {
            decoder_name = format!("{device:?} hardware");
            builder = builder.with_hardware_acceleration(*device);
        } else {
            decoder_name = "cpu (no compatible hardware device found)".into();
        }
    }
    let mut decoder = builder
        .build()
        .map_err(|e| format!("failed to open {video}: {e}"))?;

    let video_size = decoder.size();
    let (width, height) = (video_size.0 as usize, video_size.1 as usize);
    let decoder_time_base = decoder.time_base();

    let (backend, encoder_name) =
        create_backend(encoder_choice, video_out, width, height, decoder_time_base)?;

    let anaglyph_matrix = anaglyph_type_to_matrix(anaglyph_type);
    // The `video` feature implies `gpu`, so a GPU blender is always tryable
    // here; `Cpu` skips the (not free) adapter initialization entirely.
    let (mut blender, mut blender_name) = match blend {
        BlenderChoice::Cpu => (None, "CPU (rayon, CPU blending selected)".to_string()),
        BlenderChoice::Auto | BlenderChoice::Gpu => {
            match GpuBlender::new(video_size.0, video_size.1, anaglyph_matrix) {
                Ok(b) => (Some(b), "GPU (wgpu)".to_string()),
                Err(e) if blend == BlenderChoice::Gpu => {
                    return Err(format!("GPU blending requested but unavailable: {e}"))
                }
                Err(e) => (None, format!("CPU (rayon, GPU unavailable: {e})")),
            }
        }
    };

    let mut pairer = match direction {
        DirectionChoice::Fixed(d) => Pairer::fixed(d),
        DirectionChoice::Auto { reverse } => Pairer::auto(reverse),
    };
    // Auto mode measures the frame-to-frame image drift on the decode thread
    // (it overlaps with blending and the extra cost is a small downsample
    // plus a block search).
    let mut estimator = match direction {
        DirectionChoice::Fixed(_) => None,
        DirectionChoice::Auto { .. } => Some(MotionEstimator::new(width, height)),
    };

    let (first_time, first_frame) = decoder
        .decode()
        .map_err(|e| format!("failed to decode {video}: {e}"))?;
    let first_sample = estimator
        .as_mut()
        .map(|est| est.observe(raw_rgb(&first_frame)));
    pairer.push(first_time, first_frame, first_sample);

    // Decode on a background thread while this thread blends. The channel is
    // bounded (backpressure) and carries decode failures, so a broken stream
    // surfaces to the caller instead of silently truncating the output.
    let (tx, rx) = mpsc::sync_channel::<Result<(Time, Array3<u8>, Option<MotionSample>), String>>(
        DECODE_QUEUE_DEPTH,
    );
    let decode_thread = thread::spawn(move || {
        // `decode_iter` yields forever: end of stream arrives as the
        // `DecodeExhausted` sentinel, which is a normal stop, not a failure.
        for frame in decoder.decode_iter() {
            match frame {
                Ok((time, frame)) => {
                    let sample = estimator.as_mut().map(|est| est.observe(raw_rgb(&frame)));
                    if tx.send(Ok((time, frame, sample))).is_err() {
                        break; // consumer is gone
                    }
                }
                Err(video_rs::error::Error::DecodeExhausted) => break,
                Err(e) => {
                    let _ = tx.send(Err(format!("decode error: {e}")));
                    break;
                }
            }
        }
    });

    // Encode on a second background thread so CPU conversion and the encoder
    // overlap with GPU blending of the next frame. The bounded queue keeps
    // memory in flight in check.
    let (job_tx, encode_thread) = spawn_encoder(backend, (height, width));

    let mut timestamp_repair = TimestampRepair::new(decoder_time_base);
    let mut frame_count: u64 = 0;
    let mut decode_error: Option<String> = None;
    let mut reversals = Vec::new();
    let mut motion = Vec::new();
    let mut decode_done = false;
    let mut tail_flushed = false;
    let mut encode_stopped = false;
    // Stream frames through the pairer; once the decode thread is drained,
    // `finish` commits the held-back tail and the same drain loop emits it.
    while !decode_done || !tail_flushed {
        if decode_done {
            tail_flushed = true;
            (reversals, motion) = pairer.finish();
        } else {
            match rx.recv() {
                Ok(Ok((time, frame, sample))) => pairer.push(time, frame, sample),
                Ok(Err(e)) => {
                    decode_error = Some(e);
                    decode_done = true;
                }
                // `Disconnected` means the sender dropped: end of stream.
                Err(_) => decode_done = true,
            }
        }
        while pairer.ready() {
            let previous_time = pairer.front_time();
            // Scope the pixel borrows so `pop_front` can run afterwards.
            let rgb = {
                let (left, right) = pairer.front_pixels();
                let blended = match &mut blender {
                    Some(gpu) => gpu.blend(left, right),
                    None => Ok(blend_cpu(left, right, anaglyph_matrix)),
                };
                match blended {
                    Ok(rgb) => rgb,
                    Err(e) => {
                        // GPU device lost or readback failed mid-run: drop to
                        // CPU blending for this and all remaining frames
                        // rather than killing the conversion.
                        blender = None;
                        blender_name = format!("CPU (rayon, GPU blend failed mid-run: {e})");
                        blend_cpu(left, right, anaglyph_matrix)
                    }
                }
            };
            pairer.pop_front();
            let encode_time = timestamp_repair.repair(previous_time);
            if job_tx.send(Job::Frame(rgb, encode_time)).is_err() {
                // Encode thread died; its error is collected after join below.
                encode_stopped = true;
                break;
            }
            frame_count += 1;
        }
        if encode_stopped {
            break;
        }
    }
    // Releasing the receiver unwinds a decode thread blocked on a full queue.
    drop(rx);

    // Give the muxer a packet beyond the final real frame: the MP4 muxer
    // cannot derive a duration for the very last packet and ends the track at
    // its PTS, which makes strict readers drop the final frame. One duplicate
    // frame at the end lets every real frame get a duration; the duplicate
    // itself lands at the track end and is invisible during playback.
    if let Some(guard_time) = timestamp_repair.next_time() {
        let _ = job_tx.send(Job::Guard(guard_time));
    }
    drop(job_tx);

    let encode_result = match encode_thread.join() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(format!("encoding failed: {e}")),
        Err(_) => Err("encode thread panicked".into()),
    };
    // The channel receiver is gone, so the decode thread is unwinding too.
    // A panic inside the decoder (video-rs or FFmpeg bindings) drops the
    // sender just like a clean end of stream would, so the join result is
    // the only way to tell "stream ended" from "reader crashed mid-file".
    let decode_ok = decode_thread.join().is_ok();

    encode_result?;
    if let Some(e) = decode_error {
        return Err(format!(
            "decoding stopped early after {frame_count} frames: {e}"
        ));
    }
    if !decode_ok {
        return Err(format!(
            "the decoder crashed while reading {video}; \
             the file may be corrupt (run with RUST_BACKTRACE=1 for details)"
        ));
    }
    if frame_count == 0 {
        return Err(format!(
            "{video} contains fewer than two frames; nothing to blend"
        ));
    }

    Ok(ConversionReport {
        frames: frame_count,
        decoder: decoder_name,
        blender: blender_name,
        encoder: encoder_name,
        reversals,
        motion,
    })
}

/// Move a backend onto its own thread. The returned sender accepts jobs;
/// drop it to end the thread, whose `join()` yields the final result.
fn spawn_encoder(
    backend: Backend,
    shape: (usize, usize),
) -> (SyncSender<Job>, JoinHandle<Result<(), String>>) {
    let (job_tx, job_rx) = mpsc::sync_channel::<Job>(ENCODE_QUEUE_DEPTH);
    let handle = thread::spawn(move || {
        let mut backend = backend;
        let mut last_frame: Option<Vec<u8>> = None;
        for job in job_rx.iter() {
            match job {
                Job::Frame(rgb, time) => {
                    last_frame = Some(backend.encode(rgb, time, shape)?);
                }
                Job::Guard(time) => {
                    if let Some(rgb) = last_frame.take() {
                        backend.encode(rgb, time, shape)?;
                    }
                }
            }
        }
        backend.finish()
    });
    (job_tx, handle)
}

/// Create the encoding backend for the output file, along with a
/// human-readable description of the chosen backend.
fn create_backend(
    choice: EncoderChoice,
    video_out: &str,
    width: usize,
    height: usize,
    decoder_time_base: Rational,
) -> Result<(Backend, String), String> {
    let software = || -> Result<(Backend, String), String> {
        let settings = Settings::preset_h264_yuv420p(width, height, false);
        let encoder = Encoder::new(Path::new(video_out), settings)
            .map_err(|e| format!("failed to create software encoder: {}", e))?;
        Ok((
            Backend::Software(Box::new(encoder)),
            "software (libx264)".to_string(),
        ))
    };

    match choice {
        EncoderChoice::Software => software(),
        EncoderChoice::Hardware | EncoderChoice::Auto => {
            let mut reasons = Vec::new();
            for kind in HARDWARE_ORDER {
                match open_hardware(kind, video_out, width, height, decoder_time_base) {
                    Ok(ok) => return Ok(ok),
                    Err(e) => reasons.push(format!("{} unavailable: {e}", kind.display())),
                }
            }
            let tried = reasons.join("; ");
            if choice == EncoderChoice::Hardware {
                Err(format!("no hardware encoder available ({tried})"))
            } else {
                software().map(|(backend, _)| (backend, format!("software (libx264, {tried})")))
            }
        }
    }
}

/// Open one hardware encoder, reporting which one was chosen.
fn open_hardware(
    kind: HwCodec,
    video_out: &str,
    width: usize,
    height: usize,
    decoder_time_base: Rational,
) -> Result<(Backend, String), String> {
    let encoder = hwencode::open(
        kind,
        Path::new(video_out),
        width as u32,
        height as u32,
        decoder_time_base,
        EncodeOptions::default(),
    )?;
    Ok((
        Backend::Hardware(encoder),
        format!("{} ({})", kind.display(), kind.ffmpeg_name()),
    ))
}

/// CPU fallback for blending two dense RGB24 frames, parallelized with rayon.
fn blend_cpu(left: &[u8], right: &[u8], anaglyph_matrix: &[[f32; 9]; 2]) -> Vec<u8> {
    let mut out = vec![0u8; left.len()];
    out.par_chunks_mut(3)
        .zip(left.par_chunks_exact(3))
        .zip(right.par_chunks_exact(3))
        .for_each(|((out, left), right)| combine_slices(left, right, out, anaglyph_matrix));
    out
}

/// Get a contiguous RGB24 byte slice from a decoded frame.
///
/// Invariant, not input validation: video-rs hands out densely packed,
/// `Vec`-backed `Array3<u8>` frames, so `as_slice` cannot fail.
fn raw_rgb(frame: &ndarray::Array3<u8>) -> &[u8] {
    frame.as_slice().expect("decoded frames are contiguous")
}

/// Repairs missing frame timestamps.
///
/// The last packets of a stream (common with B-frames) can have no DTS, and
/// the muxer silently drops frames encoded with such a timestamp. Instead of
/// dropping the frame, synthesize its timestamp from the last valid one and
/// the observed timestamp step.
struct TimestampRepair {
    time_base: Rational,
    last_value: Option<i64>,
    step: Option<i64>,
}

impl TimestampRepair {
    fn new(time_base: Rational) -> Self {
        TimestampRepair {
            time_base,
            last_value: None,
            step: None,
        }
    }

    /// The timestamp one frame step past the last repaired one, if both the
    /// last value and an observed step are known.
    fn next_time(&self) -> Option<Time> {
        match (self.last_value, self.step) {
            (Some(last), Some(step)) => Some(Time::new(Some(last + step), self.time_base)),
            _ => None,
        }
    }

    fn repair(&mut self, time: Time) -> Time {
        if time.has_no_pts() || !time.has_value() {
            match (self.last_value, self.step) {
                (Some(last), Some(step)) => {
                    let repaired = last + step;
                    self.last_value = Some(repaired);
                    Time::new(Some(repaired), self.time_base)
                }
                _ => time,
            }
        } else {
            if let Some(value) = time.into_value() {
                if let Some(last) = self.last_value {
                    if value > last {
                        self.step = Some(value - last);
                    }
                }
                self.last_value = Some(value);
            }
            time
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use video_rs::ffmpeg::ffi;

    fn tb() -> Rational {
        Rational(1, 30)
    }

    #[test]
    fn repair_passes_valid_timestamps_through() {
        let mut repair = TimestampRepair::new(tb());
        assert_eq!(
            repair.repair(Time::new(Some(0), tb())).into_value(),
            Some(0)
        );
        assert_eq!(
            repair.repair(Time::new(Some(2), tb())).into_value(),
            Some(2)
        );
    }

    #[test]
    fn repair_synthesizes_missing_timestamps_from_observed_step() {
        let mut repair = TimestampRepair::new(tb());
        repair.repair(Time::new(Some(10), tb()));
        repair.repair(Time::new(Some(12), tb())); // observed step: 2
        assert_eq!(repair.repair(Time::new(None, tb())).into_value(), Some(14));
        assert_eq!(
            repair
                .repair(Time::new(Some(ffi::AV_NOPTS_VALUE), tb()))
                .into_value(),
            Some(16)
        );
        assert_eq!(repair.next_time().and_then(Time::into_value), Some(18));
    }

    #[test]
    fn repair_without_known_step_leaves_timestamp_untouched() {
        let mut repair = TimestampRepair::new(tb());
        repair.repair(Time::new(Some(5), tb()));
        // Only one observation so far: no step to extrapolate from.
        assert!(!repair.repair(Time::new(None, tb())).has_value());
        assert!(repair.next_time().is_none());
    }

    /// Pins the physical sign convention (see `trend_to_direction`): content
    /// drifting right means the camera moved right, so the *older* frame is
    /// the left-eye view. Regressions here would silently produce
    /// pseudoscopic output.
    #[test]
    fn trend_maps_to_eye_pairing_by_convention() {
        assert_eq!(
            trend_to_direction(Some(Trend::Increasing), false),
            VideoDirection::CounterClockwise
        );
        assert_eq!(
            trend_to_direction(Some(Trend::Decreasing), false),
            VideoDirection::Clockwise
        );
        assert_eq!(
            trend_to_direction(None, false),
            VideoDirection::Clockwise,
            "undecided motion falls back to the historical default"
        );
        // `reverse` flips every mapping, including the fallback.
        assert_eq!(
            trend_to_direction(Some(Trend::Increasing), true),
            VideoDirection::Clockwise
        );
        assert_eq!(
            trend_to_direction(Some(Trend::Decreasing), true),
            VideoDirection::CounterClockwise
        );
        assert_eq!(
            trend_to_direction(None, true),
            VideoDirection::CounterClockwise
        );
    }
}
