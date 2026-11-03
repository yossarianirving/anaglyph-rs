//! Hardware-accelerated H.264 encoding through FFmpeg's VAAPI wrapper.
//!
//! video-rs only drives software encoders, so this module talks to FFmpeg
//! directly through the `ffmpeg-next` re-export (`video_rs::ffmpeg`), using
//! raw FFI for the pieces the safe wrappers do not cover (hardware device
//! and frame contexts, the encode loop, and muxing).
//!
//! Frames are accepted as dense RGB24 slices. Each frame is converted to
//! NV12 on the CPU with libswscale, uploaded to a VAAPI surface, and fed to
//! `h264_vaapi`; packets are muxed into an MP4 written through the raw
//! `AVFormatContext` API.

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

type Result<T> = std::result::Result<T, String>;

/// Convert a negative FFmpeg error code into a message.
fn err_str(code: c_int) -> String {
    let mut buf = [0u8; 256];
    unsafe {
        ffi::av_strerror(code, buf.as_mut_ptr() as *mut c_char, buf.len());
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        format!("{} ({})", String::from_utf8_lossy(&buf[..end]), code)
    }
}

/// Check an FFmpeg return code.
fn check(code: c_int, what: &str) -> Result<()> {
    if code < 0 {
        Err(format!("{}: {}", what, err_str(code)))
    } else {
        Ok(())
    }
}

/// A VAAPI-backed H.264 encoder that muxes into an MP4 file.
///
/// Every pointer field is null unless owned, so `Drop` doubles as the
/// cleanup path for partially constructed encoders.
pub struct VaapiEncoder {
    /// Encoder codec context (may carry an open or merely allocated state).
    enc_ctx: *mut ffi::AVCodecContext,
    /// Opened muxer context (owns `pb` once `avio_open2` succeeded).
    fmt_ctx: *mut ffi::AVFormatContext,
    device_ref: *mut ffi::AVBufferRef,
    /// Buffer holding the initialized `AVHWFramesContext` pool.
    pool_ref: *mut ffi::AVBufferRef,
    /// Reusable NV12 (system memory) frame handed to the scaler.
    nv12_frame: *mut ffi::AVFrame,
    /// Reusable RGB24 view of the caller's buffer (never owns the data).
    src_frame: *mut ffi::AVFrame,
    /// Reusable VAAPI upload surface (holds its own pool reference).
    hw_frame: *mut ffi::AVFrame,
    sws: *mut ffi::SwsContext,
    packet: *mut ffi::AVPacket,
    stream_index: c_int,
    /// Time base of timestamps passed to [`Self::encode`].
    time_base: Rational,
    /// Time base the muxer picked for the stream (valid after header write).
    stream_tb: Rational,
    header_written: bool,
    trailer_written: bool,
}

// The encoder is driven from a single thread at a time (it is moved onto the
// encode thread and joined before drop), which is exactly the Send contract.
unsafe impl Send for VaapiEncoder {}

impl VaapiEncoder {
    /// Open a VAAPI encoder for `width` x `height` video writing to `path`.
    ///
    /// Timestamps passed to [`Self::encode`] are expressed in `time_base`
    /// (the source decoder's time base is the natural choice).
    pub fn new(
        path: &Path,
        width: u32,
        height: u32,
        time_base: Rational,
        options: EncodeOptions,
    ) -> Result<VaapiEncoder> {
        let mut enc = VaapiEncoder {
            enc_ctx: ptr::null_mut(),
            fmt_ctx: ptr::null_mut(),
            device_ref: ptr::null_mut(),
            pool_ref: ptr::null_mut(),
            nv12_frame: ptr::null_mut(),
            src_frame: ptr::null_mut(),
            hw_frame: ptr::null_mut(),
            sws: ptr::null_mut(),
            packet: ptr::null_mut(),
            stream_index: 0,
            time_base,
            stream_tb: Rational(0, 1),
            header_written: false,
            trailer_written: false,
        };
        enc.init(path, width, height, options)?;
        Ok(enc)
    }

    fn init(&mut self, path: &Path, width: u32, height: u32, options: EncodeOptions) -> Result<()> {
        unsafe {
            // --- hardware device ---
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

            // --- hardware frames pool ---
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

            // --- encoder ---
            let codec_name = CString::new("h264_vaapi").unwrap();
            let codec = ffi::avcodec_find_encoder_by_name(codec_name.as_ptr());
            if codec.is_null() {
                return Err("h264_vaapi encoder not available in this FFmpeg build".into());
            }
            self.enc_ctx = ffi::avcodec_alloc_context3(codec);
            if self.enc_ctx.is_null() {
                return Err("allocating encoder context: out of memory".into());
            }
            let c = &mut *self.enc_ctx;
            c.width = width as c_int;
            c.height = height as c_int;
            c.pix_fmt = ffi::AVPixelFormat::AV_PIX_FMT_VAAPI;
            c.time_base = ffi::AVRational {
                num: self.time_base.0,
                den: self.time_base.1,
            };
            c.gop_size = options.gop_size;
            // mp4 stores parameter sets in the container header.
            c.flags |= ffi::AV_CODEC_FLAG_GLOBAL_HEADER as c_int;
            c.hw_frames_ctx = ffi::av_buffer_ref(self.pool_ref);

            // Constant QP rate control; no target bitrate needed and closest
            // in spirit to x264's default CRF mode.
            let mut dict: *mut ffi::AVDictionary = ptr::null_mut();
            let configured = set_option(&mut dict, "rc_mode", "CQP")
                .and_then(|_| set_option(&mut dict, "qp", &options.qp.to_string()))
                .and_then(|_| {
                    set_option(&mut dict, "async_depth", &options.async_depth.to_string())
                })
                .and_then(|_| {
                    if options.low_power {
                        set_option(&mut dict, "low_power", "1")
                    } else {
                        Ok(())
                    }
                });
            if let Err(e) = configured {
                if !dict.is_null() {
                    ffi::av_dict_free(&mut dict);
                }
                return Err(e);
            }
            let open = ffi::avcodec_open2(self.enc_ctx, codec, &mut dict);
            if !dict.is_null() {
                // Unrecognized or unconsumed options are returned here; if we
                // got here with leftovers the option names are wrong.
                let leftover = first_dict_key(dict);
                ffi::av_dict_free(&mut dict);
                if open < 0 {
                    return Err(format!(
                        "opening h264_vaapi encoder: {} (unconsumed option: {leftover})",
                        err_str(open)
                    ));
                }
                return Err(format!("unsupported encoder options: {leftover}"));
            }
            check(open, "opening h264_vaapi encoder")?;

            // --- scaling and upload scratch ---
            self.nv12_frame = ffi::av_frame_alloc();
            self.src_frame = ffi::av_frame_alloc();
            self.hw_frame = ffi::av_frame_alloc();
            self.packet = ffi::av_packet_alloc();
            if self.nv12_frame.is_null()
                || self.src_frame.is_null()
                || self.hw_frame.is_null()
                || self.packet.is_null()
            {
                return Err("allocating frames/packet: out of memory".into());
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

            // --- muxer ---
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
                ffi::avcodec_parameters_from_context((*stream).codecpar, self.enc_ctx),
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
    }

    /// Encode one dense RGB24 frame with the given presentation timestamp,
    /// expressed in the time base passed to [`Self::new`].
    pub fn encode(&mut self, rgb: &[u8], pts: i64) -> Result<()> {
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

            // Upload NV12 -> VAAPI surface. The surface comes from the
            // frames pool; av_hwframe_get_buffer populates format/size/the
            // surface reference itself.
            let hw = self.hw_frame;
            ffi::av_frame_unref(hw);
            check(
                ffi::av_hwframe_get_buffer(self.pool_ref, hw, 0),
                "acquiring VAAPI surface from pool",
            )?;
            check(
                ffi::av_hwframe_transfer_data(hw, self.nv12_frame, 0),
                "uploading frame to GPU",
            )?;
            (*hw).pts = pts;

            // Feed the encoder; on EAGAIN drain packets and retry.
            loop {
                let ret = ffi::avcodec_send_frame(self.enc_ctx, hw);
                if ret == ffi::AVERROR(ffi::EAGAIN) {
                    self.drain_packets()?;
                    continue;
                }
                check(ret, "send_frame")?;
                break;
            }
            self.drain_packets().map(|_| ())
        }
    }

    /// Flush the encoder and finalize the container. Called automatically on
    /// drop if not invoked explicitly.
    pub fn finish(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        unsafe {
            // Bounded drain: keep pumping the NULL flush signal until the
            // encoder reports EOF and stops yielding packets. The idle
            // counter bounds the loop so a stuck driver cannot spin forever.
            let mut idle_rounds = 0;
            loop {
                let ret = ffi::avcodec_send_frame(self.enc_ctx, ptr::null());
                let flushed = ret == ffi::AVERROR_EOF;
                if ret < 0 && !flushed && ret != ffi::AVERROR(ffi::EAGAIN) {
                    return Err(format!("flush send_frame: {}", err_str(ret)));
                }
                let got = self.drain_packets()?;
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

    /// Receive and mux all pending packets. Returns whether any packet was
    /// received.
    unsafe fn drain_packets(&mut self) -> Result<bool> {
        let mut any = false;
        loop {
            let ret = ffi::avcodec_receive_packet(self.enc_ctx, self.packet);
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
}

impl Drop for VaapiEncoder {
    fn drop(&mut self) {
        unsafe {
            // Best effort: finalize the container if finish() was not called.
            if self.header_written && !self.trailer_written {
                let _ = self.finish();
            }
            if !self.fmt_ctx.is_null() && !(*self.fmt_ctx).pb.is_null() {
                ffi::avio_closep(&mut (*self.fmt_ctx).pb);
            }
            if !self.fmt_ctx.is_null() {
                ffi::avformat_free_context(self.fmt_ctx);
                self.fmt_ctx = ptr::null_mut();
            }
            if !self.hw_frame.is_null() {
                ffi::av_frame_free(&mut self.hw_frame);
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
            if !self.packet.is_null() {
                ffi::av_packet_free(&mut self.packet);
            }
            if !self.enc_ctx.is_null() {
                // Also unrefs hw_frames_ctx.
                ffi::avcodec_free_context(&mut self.enc_ctx);
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
