//! Hardware-accelerated H.264 encoding through FFmpeg's raw codec API.
//!
//! video-rs only drives software encoders, so this module talks to FFmpeg
//! directly through the `ffmpeg-next` re-export (`video_rs::ffmpeg`), using
//! raw FFI for the pieces the safe wrappers do not cover (hardware device
//! and frame contexts, the encode loop, and muxing).
//!
//! # Architecture
//!
//! The pipeline is split so that supporting a new hardware encoder means
//! implementing one trait, not duplicating the encode loop and muxer:
//!
//! - [`HwBackend`] — the per-encoder adapter: hardware device and surface
//!   lifecycle, codec-context configuration, and how a scaled NV12 frame
//!   becomes the frame the codec consumes (upload it to a surface, or pass
//!   it through when the encoder reads system memory). [`Vaapi`] implements
//!   it for `h264_vaapi`.
//! - `Driver<B: HwBackend>` (private) — the generic machinery everything
//!   backend-independent: RGB24 -> NV12 scaling, codec opening, the
//!   `send_frame`/`receive_packet` loop, and the zero-frame flush guard.
//! - [`Mp4Muxer`] (private) — the container: stream header, packet
//!   interleaving, flush, and trailer.
//!
//! [`open`] maps an [`HwCodec`] choice to a concrete `Driver` and hands
//! back a `Box<dyn HwEncoder>`, keeping callers decoupled from which
//! backend is running.
//!
//! # Adding a backend
//!
//! 1. Add a variant to [`HwCodec`] (FFmpeg codec name and display name).
//! 2. Implement [`HwBackend`]: create the device/pool in `init_device`
//!    (encoders that accept system-memory frames do nothing there), set
//!    `pix_fmt` / `hw_frames_ctx` and dictionary options in
//!    `configure_context`, and upload or pass frames through in
//!    `acquire_frame`. The unit test's `SoftwareBackend` is a minimal,
//!    runnable example of a no-device backend.
//! 3. Add the variant to the match in [`open`] and to `video`'s
//!    `HARDWARE_ORDER` (per-platform lists are fine; native first).
//!
//! Frames enter the pipeline as dense RGB24 slices and are converted to
//! NV12 on the CPU with libswscale before a backend sees them; packets are
//! muxed into an MP4 written through the raw `AVFormatContext` API.

use std::ffi::{c_char, c_int, CString};
use std::path::Path;
use std::ptr;

use video_rs::ffmpeg::ffi;
use video_rs::ffmpeg::Rational;

/// Encoder rate-control / GOP configuration.
#[derive(Clone, Copy, Debug)]
pub struct EncodeOptions {
    /// Constant QP for CQP rate control. 23 is close to x264's default CRF
    /// in perceptual terms, which keeps software/hardware comparisons fair.
    pub qp: i32,
    /// Maximum distance between keyframes.
    pub gop_size: i32,
    /// Encoder-side frame queue depth: how many frames the GPU may encode
    /// concurrently. Higher values pipeline better; the frame pool must be
    /// large enough to hold them plus the reference window.
    pub async_depth: i32,
    /// Use the low-power encoding path where the driver provides one.
    pub low_power: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            qp: 23,
            gop_size: 12,
            // Benchmarks showed throughput saturates at ~8 concurrent frames.
            async_depth: 8,
            low_power: false,
        }
    }
}

/// A hardware H.264 encoder this module can drive. Adding a backend starts
/// here; see the module docs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HwCodec {
    /// AMD/Intel GPU through VAAPI (Linux).
    Vaapi,
}

impl HwCodec {
    /// The FFmpeg encoder this variant drives.
    pub fn ffmpeg_name(self) -> &'static str {
        match self {
            HwCodec::Vaapi => "h264_vaapi",
        }
    }

    /// Short human-readable name for reports and error messages.
    pub fn display(self) -> &'static str {
        match self {
            HwCodec::Vaapi => "VAAPI",
        }
    }
}

/// The encode/finish surface a caller needs, object-safe so callers hold a
/// trait object instead of naming concrete backend types.
pub trait HwEncoder: Send {
    /// Encode one dense RGB24 frame with the given presentation timestamp,
    /// expressed in the time base passed to [`open`].
    fn encode(&mut self, rgb: &[u8], pts: i64) -> Result<()>;

    /// Flush the encoder and finalize the container. Called automatically
    /// on drop if not invoked explicitly.
    fn finish(&mut self) -> Result<()>;
}

/// Per-encoder adapter: everything about a hardware codec the generic
/// driver cannot know on its own.
///
/// All methods run inside the driver's encode thread on a single instance,
/// so implementations only need to be `Send` (the driver is moved between
/// threads), not thread-safe internally.
pub trait HwBackend: Send {
    /// The FFmpeg encoder to look up by name (must match
    /// [`HwCodec::ffmpeg_name`] for the codec this adapter belongs to).
    fn ffmpeg_name(&self) -> &'static str;

    /// Create hardware resources (devices, surface pools) that must exist
    /// before codec allocation. Called exactly once; resources stay alive
    /// until the backend is dropped. Backends whose encoders read
    /// system-memory frames have nothing to create.
    fn init_device(&mut self, width: u32, height: u32, options: &EncodeOptions) -> Result<()>;

    /// Configure the codec context before `avcodec_open2`: set the pixel
    /// format (and `hw_frames_ctx` when the encoder consumes hardware
    /// surfaces) and append codec-specific options to `dict`. The driver
    /// owns the context and frees any dictionary after opening.
    ///
    /// # Safety
    ///
    /// `ctx` and `dict` are valid pointers owned by the driver, and the
    /// context has not been opened yet.
    unsafe fn configure_context(
        &self,
        ctx: *mut ffi::AVCodecContext,
        options: &EncodeOptions,
        dict: &mut *mut ffi::AVDictionary,
    ) -> Result<()>;

    /// Turn the scaled NV12 frame into the frame to hand the codec, setting
    /// its `pts` as a side effect. Implementations either upload `nv12` to
    /// a hardware surface (which the backend owns) or return `nv12`
    /// itself when the encoder reads system memory directly.
    ///
    /// # Safety
    ///
    /// `nv12` is a valid, fully initialized NV12 frame owned by the driver.
    /// The returned frame is owned by the backend and must stay valid
    /// until the next call; the caller must not free it.
    unsafe fn acquire_frame(
        &mut self,
        nv12: *mut ffi::AVFrame,
        pts: i64,
    ) -> Result<*mut ffi::AVFrame>;
}

/// Open a hardware H.264 encoder writing to `path`.
///
/// Timestamps passed to [`HwEncoder::encode`] are expressed in `time_base`
/// (the source decoder's time base is the natural choice).
///
/// Availability is probed by opening: an encoder missing from the FFmpeg
/// build fails the codec lookup, and one whose driver is absent fails in
/// `avcodec_open2`. Both come back as ordinary errors, so callers can fall
/// back without touching the hardware.
pub fn open(
    kind: HwCodec,
    path: &Path,
    width: u32,
    height: u32,
    time_base: Rational,
    options: EncodeOptions,
) -> Result<Box<dyn HwEncoder>> {
    match kind {
        HwCodec::Vaapi => Ok(Box::new(Driver::new(
            Vaapi::new(),
            path,
            width,
            height,
            time_base,
            options,
        )?)),
    }
}

pub(crate) type Result<T> = std::result::Result<T, String>;

/// Convert a negative FFmpeg error code into a message.
pub(crate) fn err_str(code: c_int) -> String {
    let mut buf = [0u8; 256];
    unsafe {
        ffi::av_strerror(code, buf.as_mut_ptr() as *mut c_char, buf.len());
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        format!("{} ({})", String::from_utf8_lossy(&buf[..end]), code)
    }
}

/// Check an FFmpeg return code.
pub(crate) fn check(code: c_int, what: &str) -> Result<()> {
    if code < 0 {
        Err(format!("{}: {}", what, err_str(code)))
    } else {
        Ok(())
    }
}

/// The VAAPI backend: an owned device, an NV12-compatible surface pool,
/// and one reusable upload surface.
///
/// Every pointer field is null unless owned, so `Drop` doubles as the
/// cleanup path for partially initialized state.
pub struct Vaapi {
    device_ref: *mut ffi::AVBufferRef,
    /// Buffer holding the initialized `AVHWFramesContext` pool.
    pool_ref: *mut ffi::AVBufferRef,
    /// Reusable VAAPI upload surface (holds its own pool reference).
    hw_frame: *mut ffi::AVFrame,
}

// The backend is driven from a single thread at a time (the driver that
// owns it is moved onto the encode thread and joined before drop).
unsafe impl Send for Vaapi {}

impl Vaapi {
    fn new() -> Vaapi {
        Vaapi {
            device_ref: ptr::null_mut(),
            pool_ref: ptr::null_mut(),
            hw_frame: ptr::null_mut(),
        }
    }
}

impl HwBackend for Vaapi {
    fn ffmpeg_name(&self) -> &'static str {
        HwCodec::Vaapi.ffmpeg_name()
    }

    fn init_device(&mut self, width: u32, height: u32, options: &EncodeOptions) -> Result<()> {
        unsafe {
            check(
                ffi::av_hwdevice_ctx_create(
                    &mut self.device_ref,
                    ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                    ptr::null(),
                    ptr::null_mut(),
                    0,
                ),
                "creating VAAPI device context",
            )?;

            self.pool_ref = ffi::av_hwframe_ctx_alloc(self.device_ref);
            if self.pool_ref.is_null() {
                return Err("allocating hardware frames context: out of memory".into());
            }
            let frames = (*self.pool_ref).data as *mut ffi::AVHWFramesContext;
            (*frames).format = ffi::AVPixelFormat::AV_PIX_FMT_VAAPI;
            (*frames).sw_format = ffi::AVPixelFormat::AV_PIX_FMT_NV12;
            (*frames).width = width as c_int;
            (*frames).height = height as c_int;
            // Surfaces in flight = async queue + reference window + slack.
            (*frames).initial_pool_size = options.async_depth + 16;
            check(
                ffi::av_hwframe_ctx_init(self.pool_ref),
                "initializing hardware frames pool",
            )?;

            self.hw_frame = ffi::av_frame_alloc();
            if self.hw_frame.is_null() {
                return Err("allocating upload surface: out of memory".into());
            }
            Ok(())
        }
    }

    unsafe fn configure_context(
        &self,
        ctx: *mut ffi::AVCodecContext,
        options: &EncodeOptions,
        dict: &mut *mut ffi::AVDictionary,
    ) -> Result<()> {
        let c = &mut *ctx;
        c.pix_fmt = ffi::AVPixelFormat::AV_PIX_FMT_VAAPI;
        c.hw_frames_ctx = ffi::av_buffer_ref(self.pool_ref);

        // Constant QP rate control; no target bitrate needed and closest
        // in spirit to x264's default CRF mode.
        set_option(dict, "rc_mode", "CQP")
            .and_then(|_| set_option(dict, "qp", &options.qp.to_string()))
            .and_then(|_| set_option(dict, "async_depth", &options.async_depth.to_string()))
            .and_then(|_| {
                if options.low_power {
                    set_option(dict, "low_power", "1")
                } else {
                    Ok(())
                }
            })
    }

    unsafe fn acquire_frame(
        &mut self,
        nv12: *mut ffi::AVFrame,
        pts: i64,
    ) -> Result<*mut ffi::AVFrame> {
        // The surface comes from the frames pool; av_hwframe_get_buffer
        // populates format/size/the surface reference itself.
        ffi::av_frame_unref(self.hw_frame);
        check(
            ffi::av_hwframe_get_buffer(self.pool_ref, self.hw_frame, 0),
            "acquiring VAAPI surface from pool",
        )?;
        check(
            ffi::av_hwframe_transfer_data(self.hw_frame, nv12, 0),
            "uploading frame to GPU",
        )?;
        (*self.hw_frame).pts = pts;
        Ok(self.hw_frame)
    }
}

impl Drop for Vaapi {
    fn drop(&mut self) {
        unsafe {
            if !self.hw_frame.is_null() {
                ffi::av_frame_free(&mut self.hw_frame);
            }
            if !self.pool_ref.is_null() {
                ffi::av_buffer_unref(&mut self.pool_ref);
            }
            if !self.device_ref.is_null() {
                ffi::av_buffer_unref(&mut self.device_ref);
            }
        }
    }
}

/// The MP4 container: one video stream written through the raw
/// `AVFormatContext` API, plus the packet receive/rescale/interleave loop.
///
/// Every pointer field is null unless owned, so `Drop` doubles as the
/// cleanup path for partially opened state.
struct Mp4Muxer {
    /// Opened muxer context (owns `pb` once `avio_open2` succeeded).
    fmt_ctx: *mut ffi::AVFormatContext,
    packet: *mut ffi::AVPacket,
    stream_index: c_int,
    /// Time base of timestamps passed to the encoder.
    time_base: Rational,
    /// Time base the muxer picked for the stream (valid after header write).
    stream_tb: Rational,
    header_written: bool,
    trailer_written: bool,
}

impl Mp4Muxer {
    fn new(time_base: Rational) -> Mp4Muxer {
        Mp4Muxer {
            fmt_ctx: ptr::null_mut(),
            packet: ptr::null_mut(),
            stream_index: 0,
            time_base,
            stream_tb: Rational(0, 1),
            header_written: false,
            trailer_written: false,
        }
    }

    /// Create the container for `path` and write the stream header taken
    /// from the (already opened) encoder context.
    unsafe fn open(&mut self, path: &Path, enc_ctx: *mut ffi::AVCodecContext) -> Result<()> {
        self.packet = ffi::av_packet_alloc();
        if self.packet.is_null() {
            return Err("allocating packet: out of memory".into());
        }
        let c_path = CString::new(path.as_os_str().to_string_lossy().as_bytes())
            .map_err(|_| "output path contains a NUL byte".to_string())?;
        check(
            ffi::avformat_alloc_output_context2(
                &mut self.fmt_ctx,
                ptr::null_mut(),
                ptr::null(),
                c_path.as_ptr(),
            ),
            &format!("creating muxer for {:?}", path),
        )?;
        if self.fmt_ctx.is_null() {
            return Err(format!("no muxer for {:?}", path));
        }
        let stream = ffi::avformat_new_stream(self.fmt_ctx, ptr::null());
        if stream.is_null() {
            return Err("adding stream to muxer: out of memory".into());
        }
        self.stream_index = (*stream).index;
        check(
            ffi::avcodec_parameters_from_context((*stream).codecpar, enc_ctx),
            "copying codec parameters",
        )?;
        check(
            ffi::avio_open2(
                &mut (*self.fmt_ctx).pb,
                c_path.as_ptr(),
                ffi::AVIO_FLAG_WRITE,
                ptr::null(),
                ptr::null_mut(),
            ),
            &format!("opening {:?}", path),
        )?;
        check(
            ffi::avformat_write_header(self.fmt_ctx, ptr::null_mut()),
            "writing container header",
        )?;
        self.header_written = true;
        self.stream_tb = {
            let tb = (*stream).time_base;
            Rational(tb.num, tb.den)
        };
        Ok(())
    }

    /// Receive and mux all pending packets. Returns whether any packet was
    /// received.
    unsafe fn drain(&mut self, enc_ctx: *mut ffi::AVCodecContext) -> Result<bool> {
        let mut any = false;
        loop {
            let ret = ffi::avcodec_receive_packet(enc_ctx, self.packet);
            if ret == ffi::AVERROR_EOF || ret == ffi::AVERROR(ffi::EAGAIN) {
                return Ok(any);
            }
            check(ret, "receive_packet")?;
            any = true;
            let pkt = &mut *self.packet;
            pkt.stream_index = self.stream_index;
            ffi::av_packet_rescale_ts(
                self.packet,
                ffi::AVRational {
                    num: self.time_base.0,
                    den: self.time_base.1,
                },
                ffi::AVRational {
                    num: self.stream_tb.0,
                    den: self.stream_tb.1,
                },
            );
            // av_interleaved_write_frame takes ownership of the packet,
            // including on failure.
            check(
                ffi::av_interleaved_write_frame(self.fmt_ctx, self.packet),
                "writing packet",
            )?;
        }
    }

    /// Flush the encoder and write the trailer. The caller guarantees at
    /// least one frame was sent to the encoder (never flush a codec that
    /// saw nothing — see `Driver::finish`).
    unsafe fn flush_and_write_trailer(&mut self, enc_ctx: *mut ffi::AVCodecContext) -> Result<()> {
        // Bounded drain: keep pumping the NULL flush signal until the
        // encoder reports EOF and stops yielding packets. The idle counter
        // bounds the loop so a stuck driver cannot spin forever.
        let mut idle_rounds = 0;
        loop {
            let ret = ffi::avcodec_send_frame(enc_ctx, ptr::null());
            let flushed = ret == ffi::AVERROR_EOF;
            if ret < 0 && !flushed && ret != ffi::AVERROR(ffi::EAGAIN) {
                return Err(format!("flush send_frame: {}", err_str(ret)));
            }
            let got = self.drain(enc_ctx)?;
            if flushed && !got {
                break;
            }
            if got {
                idle_rounds = 0;
            } else {
                idle_rounds += 1;
                if idle_rounds > 2 {
                    return Err("encoder made no progress while flushing".into());
                }
            }
        }
        if self.header_written {
            check(ffi::av_write_trailer(self.fmt_ctx), "writing trailer")?;
            self.trailer_written = true;
        }
        Ok(())
    }
}

impl Drop for Mp4Muxer {
    fn drop(&mut self) {
        unsafe {
            if !self.fmt_ctx.is_null() && !(*self.fmt_ctx).pb.is_null() {
                ffi::avio_closep(&mut (*self.fmt_ctx).pb);
            }
            if !self.fmt_ctx.is_null() {
                ffi::avformat_free_context(self.fmt_ctx);
                self.fmt_ctx = ptr::null_mut();
            }
            if !self.packet.is_null() {
                ffi::av_packet_free(&mut self.packet);
            }
        }
    }
}

/// The generic encode pipeline driving any [`HwBackend`]: scaling, codec
/// opening, the send/receive loop, and the finish protocol. All the
/// backend-specific decisions go through the three trait hooks.
///
/// Every pointer field is null unless owned, so `Drop` doubles as the
/// cleanup path for partially constructed drivers.
struct Driver<B: HwBackend> {
    backend: B,
    /// Encoder codec context (may carry an open or merely allocated state).
    enc_ctx: *mut ffi::AVCodecContext,
    /// Reusable NV12 (system memory) frame handed to the scaler and the
    /// backend.
    nv12_frame: *mut ffi::AVFrame,
    /// Reusable RGB24 view of the caller's buffer (never owns the data).
    src_frame: *mut ffi::AVFrame,
    sws: *mut ffi::SwsContext,
    muxer: Mp4Muxer,
    /// Time base of timestamps passed to [`HwEncoder::encode`].
    time_base: Rational,
    /// Frames successfully submitted to the encoder. Flushing an encoder
    /// that never received a frame crashes FFmpeg's h264_vaapi (its
    /// internal state is only initialized on the first frame), so `finish`
    /// must know whether there is anything to flush. The same skip is
    /// correct for every hardware encoder: nothing was encoded, so there
    /// is nothing to drain.
    frames_sent: usize,
}

// The driver is driven from a single thread at a time (it is moved onto
// the encode thread and joined before drop), which is exactly the Send
// contract. The backend's Send bound covers its own pointers.
unsafe impl<B: HwBackend> Send for Driver<B> {}

impl<B: HwBackend> Driver<B> {
    fn new(
        backend: B,
        path: &Path,
        width: u32,
        height: u32,
        time_base: Rational,
        options: EncodeOptions,
    ) -> Result<Driver<B>> {
        let mut enc = Driver {
            backend,
            enc_ctx: ptr::null_mut(),
            nv12_frame: ptr::null_mut(),
            src_frame: ptr::null_mut(),
            sws: ptr::null_mut(),
            muxer: Mp4Muxer::new(time_base),
            time_base,
            frames_sent: 0,
        };
        enc.init(path, width, height, options)?;
        Ok(enc)
    }

    fn init(&mut self, path: &Path, width: u32, height: u32, options: EncodeOptions) -> Result<()> {
        self.backend.init_device(width, height, &options)?;

        unsafe {
            // --- encoder ---
            let codec_name = CString::new(self.backend.ffmpeg_name()).unwrap();
            let codec = ffi::avcodec_find_encoder_by_name(codec_name.as_ptr());
            if codec.is_null() {
                return Err(format!(
                    "{} encoder not available in this FFmpeg build",
                    self.backend.ffmpeg_name()
                ));
            }
            self.enc_ctx = ffi::avcodec_alloc_context3(codec);
            if self.enc_ctx.is_null() {
                return Err("allocating encoder context: out of memory".into());
            }
            let c = &mut *self.enc_ctx;
            c.width = width as c_int;
            c.height = height as c_int;
            c.time_base = ffi::AVRational {
                num: self.time_base.0,
                den: self.time_base.1,
            };
            c.gop_size = options.gop_size;
            // mp4 stores parameter sets in the container header.
            c.flags |= ffi::AV_CODEC_FLAG_GLOBAL_HEADER as c_int;

            let mut dict: *mut ffi::AVDictionary = ptr::null_mut();
            if let Err(e) = self
                .backend
                .configure_context(self.enc_ctx, &options, &mut dict)
            {
                if !dict.is_null() {
                    ffi::av_dict_free(&mut dict);
                }
                return Err(e);
            }
            let open_result = ffi::avcodec_open2(self.enc_ctx, codec, &mut dict);
            if !dict.is_null() {
                // Unrecognized or unconsumed options are returned here; if we
                // got here with leftovers the option names are wrong.
                let leftover = first_dict_key(dict);
                ffi::av_dict_free(&mut dict);
                if open_result < 0 {
                    return Err(format!(
                        "opening {} encoder: {} (unconsumed option: {leftover})",
                        self.backend.ffmpeg_name(),
                        err_str(open_result)
                    ));
                }
                return Err(format!("unsupported encoder options: {leftover}"));
            }
            check(
                open_result,
                &format!("opening {} encoder", self.backend.ffmpeg_name()),
            )?;

            // --- scaling scratch ---
            self.nv12_frame = ffi::av_frame_alloc();
            self.src_frame = ffi::av_frame_alloc();
            if self.nv12_frame.is_null() || self.src_frame.is_null() {
                return Err("allocating frames: out of memory".into());
            }
            (*self.src_frame).format = ffi::AVPixelFormat::AV_PIX_FMT_RGB24 as c_int;
            (*self.src_frame).width = width as c_int;
            (*self.src_frame).height = height as c_int;
            (*self.src_frame).linesize[0] = width as c_int * 3;
            self.sws = ffi::sws_getContext(
                width as c_int,
                height as c_int,
                ffi::AVPixelFormat::AV_PIX_FMT_RGB24,
                width as c_int,
                height as c_int,
                ffi::AVPixelFormat::AV_PIX_FMT_NV12,
                ffi::SwsFlags::SWS_BILINEAR as c_int | ffi::SwsFlags::SWS_ACCURATE_RND as c_int,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
            );
            if self.sws.is_null() {
                return Err("creating rgb24 -> nv12 scaler".into());
            }
        }

        // --- muxer ---
        // Partial failures inside `open` are cleaned up by the muxer's own
        // Drop; the driver only adopts a fully opened one.
        let mut muxer = Mp4Muxer::new(self.time_base);
        unsafe { muxer.open(path, self.enc_ctx)? };
        self.muxer = muxer;
        Ok(())
    }
}

impl<B: HwBackend> HwEncoder for Driver<B> {
    fn encode(&mut self, rgb: &[u8], pts: i64) -> Result<()> {
        let (w, h) = unsafe {
            let c = &*self.enc_ctx;
            (c.width as usize, c.height as usize)
        };
        if rgb.len() != w * h * 3 {
            return Err(format!(
                "frame size mismatch: got {} bytes, expected {}",
                rgb.len(),
                w * h * 3
            ));
        }

        unsafe {
            // RGB24 (caller buffer) -> NV12 scratch frame.
            (*self.src_frame).data[0] = rgb.as_ptr() as *mut u8;
            check(
                ffi::sws_scale_frame(self.sws, self.nv12_frame, self.src_frame),
                "rgb24 -> nv12 conversion",
            )?;

            // The backend decides whether the encoder gets an uploaded
            // surface or the NV12 frame itself.
            let input = self.backend.acquire_frame(self.nv12_frame, pts)?;

            // Feed the encoder; on EAGAIN drain packets and retry.
            loop {
                let ret = ffi::avcodec_send_frame(self.enc_ctx, input);
                if ret == ffi::AVERROR(ffi::EAGAIN) {
                    self.muxer.drain(self.enc_ctx)?;
                    continue;
                }
                check(ret, "send_frame")?;
                break;
            }
            self.frames_sent += 1;
            self.muxer.drain(self.enc_ctx).map(|_| ())
        }
    }

    fn finish(&mut self) -> Result<()> {
        if self.muxer.trailer_written {
            return Ok(());
        }
        if self.frames_sent == 0 {
            // Nothing was ever encoded: there is no pipeline to flush, and
            // FFmpeg's h264_vaapi segfaults when a NULL flush reaches an
            // encoder that never saw a frame. Leave the header-only file
            // behind (the caller's run failed and removes it) and mark the
            // trailer handled so `Drop` does not retry the flush.
            self.muxer.trailer_written = true;
            return Ok(());
        }
        unsafe { self.muxer.flush_and_write_trailer(self.enc_ctx) }
    }
}

impl<B: HwBackend> Drop for Driver<B> {
    fn drop(&mut self) {
        unsafe {
            // Best effort: finalize the container if finish() was not
            // called. Requires the encoder context, so it runs before the
            // (automatic) drops of the backend and muxer fields.
            if self.muxer.header_written && !self.muxer.trailer_written {
                let _ = self.finish();
            }
            if !self.src_frame.is_null() {
                // Only borrows caller memory; freeing the frame is safe.
                ffi::av_frame_free(&mut self.src_frame);
            }
            if !self.nv12_frame.is_null() {
                ffi::av_frame_free(&mut self.nv12_frame);
            }
            if !self.sws.is_null() {
                ffi::sws_freeContext(self.sws);
                self.sws = ptr::null_mut();
            }
            if !self.enc_ctx.is_null() {
                // Also unrefs hw_frames_ctx.
                ffi::avcodec_free_context(&mut self.enc_ctx);
            }
        }
    }
}

/// Set one string option on an FFmpeg dictionary.
fn set_option(dict: &mut *mut ffi::AVDictionary, key: &str, value: &str) -> Result<()> {
    let key = CString::new(key).map_err(|_| "option key contains NUL")?;
    let value = CString::new(value).map_err(|_| "option value contains NUL")?;
    let code = unsafe { ffi::av_dict_set(dict, key.as_ptr(), value.as_ptr(), 0) };
    check(code, "setting encoder option")
}

/// Key of the first entry in a dictionary, for error reporting.
fn first_dict_key(dict: *mut ffi::AVDictionary) -> String {
    unsafe {
        let entry = ffi::av_dict_get(dict, ptr::null(), ptr::null(), 0);
        if entry.is_null() || (*entry).key.is_null() {
            return "(unknown)".into();
        }
        std::ffi::CStr::from_ptr((*entry).key)
            .to_string_lossy()
            .into_owned()
    }
}

/// A minimal no-device backend that encodes the NV12 frames the scaler
/// produces with `libx264` — exactly the shape an NVENC or VideoToolbox
/// backend would take (see the module docs). It exercises the whole generic
/// pipeline, mux included, on machines without any GPU encoder, and doubles
/// as a video-only fixture generator for tests in other modules.
#[cfg(test)]
pub(crate) struct SoftwareBackend;

#[cfg(test)]
unsafe impl Send for SoftwareBackend {}

#[cfg(test)]
impl HwBackend for SoftwareBackend {
    fn ffmpeg_name(&self) -> &'static str {
        "libx264"
    }

    fn init_device(&mut self, _width: u32, _height: u32, _options: &EncodeOptions) -> Result<()> {
        Ok(())
    }

    unsafe fn configure_context(
        &self,
        ctx: *mut ffi::AVCodecContext,
        _options: &EncodeOptions,
        _dict: &mut *mut ffi::AVDictionary,
    ) -> Result<()> {
        (*ctx).pix_fmt = ffi::AVPixelFormat::AV_PIX_FMT_NV12;
        Ok(())
    }

    unsafe fn acquire_frame(
        &mut self,
        nv12: *mut ffi::AVFrame,
        pts: i64,
    ) -> Result<*mut ffi::AVFrame> {
        (*nv12).pts = pts;
        Ok(nv12)
    }
}

/// Whether the `libx264` encoder exists in this FFmpeg build (tests skip
/// quietly on stripped builds).
#[cfg(test)]
pub(crate) fn software_encoder_available() -> bool {
    let name = CString::new("libx264").unwrap();
    unsafe { !ffi::avcodec_find_encoder_by_name(name.as_ptr()).is_null() }
}

/// Write a tiny flat-grey video-only MP4 (`frames` frames at 25 fps), used
/// as the encode target in tests that exercise container-level behavior.
#[cfg(test)]
pub(crate) fn make_test_video(path: &Path, frames: i64) -> Result<()> {
    let mut enc = Driver::new(
        SoftwareBackend,
        path,
        16,
        16,
        Rational(1, 25),
        EncodeOptions::default(),
    )?;
    for pts in 0..frames {
        enc.encode(&[7u8; 16 * 16 * 3], pts)?;
    }
    enc.finish()?;
    drop(enc);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_names() {
        assert_eq!(HwCodec::Vaapi.ffmpeg_name(), "h264_vaapi");
        assert_eq!(HwCodec::Vaapi.display(), "VAAPI");
    }

    #[test]
    fn driver_pipeline_with_system_memory_backend() {
        if !software_encoder_available() {
            return;
        }
        let dir = std::env::temp_dir().join("anaglyph-hwencode-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("system-memory.mp4");
        let _ = std::fs::remove_file(&path);

        make_test_video(&path, 4).unwrap();

        // A finished, non-trivial mp4 came out of the generic pipeline.
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000, "output too small: {}", bytes.len());
        assert_eq!(&bytes[4..8], b"ftyp");
    }

    #[test]
    fn empty_run_finishes_without_flushing() {
        // The guard against crashing an encoder that never saw a frame.
        if !software_encoder_available() {
            return;
        }
        let dir = std::env::temp_dir().join("anaglyph-hwencode-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty-run.mp4");
        let _ = std::fs::remove_file(&path);

        let mut enc = Driver::new(
            SoftwareBackend,
            &path,
            16,
            16,
            Rational(1, 25),
            EncodeOptions::default(),
        )
        .unwrap();
        enc.finish().unwrap();
        // Dropping after a finished empty run must not retry the flush.
        drop(enc);
    }
}
