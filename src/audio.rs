//! Carry the source audio track into the anaglyph output video.
//!
//! The encoders (video-rs's `Encoder` and the [`crate::hwencode`] backends)
//! write video-only MP4s, so the audio is attached as a post-encode remux
//! pass: the finished video file and the source are opened together, the
//! video stream is stream-copied unchanged, and the source audio track is
//! written alongside it into a temp file that replaces the output when it
//! completes.
//!
//! Two ways the audio lands in the container:
//!
//! - **Stream copy** when the MP4 muxer accepts the source codec (AAC, and
//!   Opus on recent FFmpeg builds): lossless, and as fast as a file copy.
//! - **AAC transcode** otherwise (Vorbis, PCM, older FFmpeg builds that mark
//!   the muxing experimental): decode with the source's decoder, resample to
//!   48 kHz fltp with libswresample, encode with the native AAC encoder.
//!
//! The pass is best-effort by design: any failure leaves the video-only
//! output untouched and is reported to the user rather than failing an
//! otherwise complete conversion. A source without audio leaves the output
//! byte-for-byte untouched.

use std::collections::VecDeque;
use std::ffi::{c_float, c_int, CString};
use std::path::{Path, PathBuf};
use std::ptr;

use video_rs::ffmpeg::ffi;

use crate::hwencode::{check, err_str, Result};

/// Sample rate used when audio must be transcoded to AAC.
const AAC_SAMPLE_RATE: c_int = 48_000;
/// Bit rate for stereo AAC output (mono gets half).
const AAC_BIT_RATE: i64 = 128_000;
/// `FF_COMPLIANCE_NORMAL` from `libavcodec/defs.h`: the compliance level at
/// which a muxer accepts a codec unconditionally.
const COMPLIANCE_NORMAL: c_int = 0;
/// Extra samples allocated in the resample scratch beyond the expected
/// output. swresample's direct (no-copy) path writes well past `out_count`
/// — measured at ~1088 samples beyond on FFmpeg 9.0.2, regardless of the
/// returned count — treating the space as its own padding; an `AVFrame`
/// buffer's 64-byte tail is nowhere near enough, and the overflow smashes
/// malloc metadata. This slack absorbs the write-ahead with room to spare.
const RESAMPLE_SLACK: usize = 2048;
/// Initial scratch capacity in samples; large enough for most decoders
/// (AAC 1024, MP3 1152, most PCM chunkings), grown on demand for the rest
/// (Opus buffers up to 5760).
const RESAMPLE_INITIAL_CAPACITY: usize = 4096;
/// `FFERRTAG` from `libavutil/error.h`: pack four tag bytes into a negative
/// FFmpeg error code.
const fn fferr_tag(a: u8, b: u8, c: u8, d: u8) -> c_int {
    -((a as u32 | (b as u32) << 8 | (c as u32) << 16 | (d as u32) << 24) as i32)
}
/// `AVERROR_STREAM_NOT_FOUND`: `av_find_best_stream`'s "no stream of that
/// type here" code, as distinct from a real failure. (The bindings do not
/// export the macro, so rebuild it from the tag bytes.)
const STREAM_NOT_FOUND: c_int = fferr_tag(0xF8, b'S', b'T', b'R');

/// Attach the audio track of `source` to the video-only MP4 at `video_out`,
/// replacing the file in place. Returns a short human-readable description
/// of the outcome for the conversion report.
///
/// `Err` means audio was dropped and the video file was left untouched.
pub fn attach_audio(source: &str, video_out: &Path) -> Result<String> {
    let Some((codec_id, track_count)) = probe_audio(source)? else {
        return Ok("none (source has no audio track)".to_string());
    };
    let codec = codec_name(codec_id);
    let suffix = if track_count > 1 {
        format!("; 1 of {track_count} tracks")
    } else {
        String::new()
    };

    if codec_supported_in_mp4(codec_id) {
        if attach_once(video_out, source, Mode::Copy).is_ok() {
            return Ok(format!("{codec} (stream copy{suffix})"));
        }
        // A copy can still fail at header time on muxers whose support
        // query said yes; fall through to the transcode.
        let reason = format!("{codec} could not be stream-copied");
        return attach_transcoded(video_out, source, &codec, &suffix, &reason);
    }
    let reason = format!("{codec} is not supported in MP4");
    attach_transcoded(video_out, source, &codec, &suffix, &reason)
}

fn attach_transcoded(
    video_out: &Path,
    source: &str,
    codec: &str,
    suffix: &str,
    reason: &str,
) -> Result<String> {
    match attach_once(video_out, source, Mode::Transcode) {
        Ok(()) => Ok(format!("aac (transcoded from {codec}{suffix}; {reason})")),
        Err(e) => Err(format!("{reason}, and the AAC fallback failed: {e}")),
    }
}

/// How the audio reaches the output container.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Copy,
    Transcode,
}

/// Open the source just far enough to find its audio configuration.
/// Returns `(codec, number of audio tracks)`, or `None` for a silent source.
fn probe_audio(source: &str) -> Result<Option<(ffi::AVCodecID, usize)>> {
    let input = Input::open(Path::new(source))?;
    let idx = unsafe { find_audio_stream(input.fmt) }?;
    let Some(idx) = idx else { return Ok(None) };
    unsafe {
        let stream = *(*input.fmt).streams.add(idx as usize);
        let codec_id = (*(*stream).codecpar).codec_id;
        let count = (0..(*input.fmt).nb_streams)
            .filter(|&i| {
                let stream = *(*input.fmt).streams.add(i as usize);
                (*(*stream).codecpar).codec_type == ffi::AVMediaType::AVMEDIA_TYPE_AUDIO
            })
            .count();
        Ok(Some((codec_id, count)))
    }
}

/// Whether the MP4 muxer accepts `codec_id` at normal compliance — asked of
/// the muxer itself rather than a hand-kept list.
fn codec_supported_in_mp4(codec_id: ffi::AVCodecID) -> bool {
    let name = CString::new("mp4").unwrap();
    unsafe {
        let oformat = ffi::av_guess_format(name.as_ptr(), ptr::null(), ptr::null());
        !oformat.is_null() && ffi::avformat_query_codec(oformat, codec_id, COMPLIANCE_NORMAL) == 1
    }
}

/// The actual remux: write video + audio to a temp file next to
/// `video_out`, then rename it over the output. A failed attempt deletes
/// only the temp file it created.
fn attach_once(video_out: &Path, source: &str, mode: Mode) -> Result<()> {
    let tmp = temp_path(video_out);
    let _ = std::fs::remove_file(&tmp);
    match remux(video_out, source, mode, &tmp) {
        Ok(()) => match std::fs::rename(&tmp, video_out) {
            Ok(()) => Ok(()),
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                Err(format!(
                    "replacing {video_out:?} with the remuxed file: {e}"
                ))
            }
        },
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn temp_path(video_out: &Path) -> PathBuf {
    let mut name = video_out.as_os_str().to_os_string();
    name.push(".audio.tmp");
    PathBuf::from(name)
}

/// Find the best audio stream: `Ok(None)` when the file has none at all
/// (as opposed to a read failure).
unsafe fn find_audio_stream(fmt: *mut ffi::AVFormatContext) -> Result<Option<c_int>> {
    let idx = unsafe {
        ffi::av_find_best_stream(
            fmt,
            ffi::AVMediaType::AVMEDIA_TYPE_AUDIO,
            -1,
            -1,
            ptr::null_mut(),
            0,
        )
    };
    if idx >= 0 {
        Ok(Some(idx))
    } else if idx == STREAM_NOT_FOUND {
        Ok(None)
    } else {
        Err(format!("finding audio stream: {}", err_str(idx)))
    }
}

/// Find the best video stream (required to exist).
unsafe fn find_video_stream(fmt: *mut ffi::AVFormatContext) -> Result<c_int> {
    let idx = unsafe {
        ffi::av_find_best_stream(
            fmt,
            ffi::AVMediaType::AVMEDIA_TYPE_VIDEO,
            -1,
            -1,
            ptr::null_mut(),
            0,
        )
    };
    if idx >= 0 {
        Ok(idx)
    } else {
        Err(format!("finding video stream in output: {}", err_str(idx)))
    }
}

/// An opened input container; Drop closes it.
struct Input {
    fmt: *mut ffi::AVFormatContext,
}

// Each `Input` is independently owned and driven from one thread at a time
// (the remux runs inline on the caller's thread).
unsafe impl Send for Input {}

impl Input {
    fn open(path: &Path) -> Result<Input> {
        let c_path = CString::new(path.as_os_str().to_string_lossy().as_bytes())
            .map_err(|_| "path contains a NUL byte".to_string())?;
        let mut fmt: *mut ffi::AVFormatContext = ptr::null_mut();
        unsafe {
            check(
                ffi::avformat_open_input(
                    &mut fmt,
                    c_path.as_ptr(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                ),
                &format!("opening {path:?}"),
            )?;
            // `avformat_open_input` frees the context on failure, so `fmt`
            // is either valid or null here; Drop tolerates both.
            check(
                ffi::avformat_find_stream_info(fmt, ptr::null_mut()),
                &format!("reading stream info from {path:?}"),
            )?;
        }
        Ok(Input { fmt })
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        unsafe {
            if !self.fmt.is_null() {
                ffi::avformat_close_input(&mut self.fmt);
            }
        }
    }
}

/// Read the next packet of `stream_idx`, discarding packets of other
/// streams. `Ok(None)` is end of stream; the packet is owned by the caller.
fn read_next_packet(
    fmt: *mut ffi::AVFormatContext,
    stream_idx: c_int,
) -> Result<Option<*mut ffi::AVPacket>> {
    loop {
        let mut packet = unsafe { ffi::av_packet_alloc() };
        if packet.is_null() {
            return Err("allocating packet: out of memory".into());
        }
        let ret = unsafe { ffi::av_read_frame(fmt, packet) };
        if ret == ffi::AVERROR_EOF {
            unsafe { ffi::av_packet_free(&mut packet) };
            return Ok(None);
        }
        if ret < 0 {
            unsafe { ffi::av_packet_free(&mut packet) };
            return Err(format!("reading input packet: {}", err_str(ret)));
        }
        if unsafe { (*packet).stream_index } == stream_idx {
            return Ok(Some(packet));
        }
        unsafe { ffi::av_packet_free(&mut packet) };
    }
}

/// An output MP4 holding one copied video stream and one audio stream.
/// Drop closes the file; a container that never got its trailer is simply
/// discarded (the caller deletes the temp file).
struct Remuxer {
    fmt: *mut ffi::AVFormatContext,
    /// Output stream indices.
    video_idx: c_int,
    audio_idx: c_int,
    /// Output stream time bases, valid after `write_header`.
    video_tb: ffi::AVRational,
    audio_tb: ffi::AVRational,
}

unsafe impl Send for Remuxer {}

impl Remuxer {
    fn open(path: &Path) -> Result<Remuxer> {
        let c_path = CString::new(path.as_os_str().to_string_lossy().as_bytes())
            .map_err(|_| "path contains a NUL byte".to_string())?;
        let mut fmt: *mut ffi::AVFormatContext = ptr::null_mut();
        unsafe {
            check(
                ffi::avformat_alloc_output_context2(
                    &mut fmt,
                    ptr::null_mut(),
                    // The temp name has no mp4 extension; name the muxer
                    // explicitly.
                    c"mp4".as_ptr(),
                    c_path.as_ptr(),
                ),
                &format!("creating remux muxer for {path:?}"),
            )?;
            if fmt.is_null() {
                return Err(format!("no mp4 muxer for {path:?}"));
            }
            check(
                ffi::avio_open2(
                    &mut (*fmt).pb,
                    c_path.as_ptr(),
                    ffi::AVIO_FLAG_WRITE,
                    ptr::null(),
                    ptr::null_mut(),
                ),
                &format!("opening {path:?}"),
            )?;
        }
        Ok(Remuxer {
            fmt,
            video_idx: -1,
            audio_idx: -1,
            video_tb: ffi::AVRational { num: 0, den: 1 },
            audio_tb: ffi::AVRational { num: 0, den: 1 },
        })
    }

    /// Add a stream carrying an exact copy of `par`.
    unsafe fn add_stream(&mut self, par: *const ffi::AVCodecParameters) -> Result<c_int> {
        let stream = unsafe { ffi::avformat_new_stream(self.fmt, ptr::null()) };
        if stream.is_null() {
            return Err("adding stream to remuxer: out of memory".into());
        }
        check(
            ffi::avcodec_parameters_copy((*stream).codecpar, par),
            "copying codec parameters",
        )?;
        Ok(unsafe { (*stream).index })
    }

    unsafe fn write_header(&mut self) -> Result<()> {
        check(
            ffi::avformat_write_header(self.fmt, ptr::null_mut()),
            "writing remux header",
        )?;
        self.video_tb = stream_tb(self.fmt, self.video_idx);
        self.audio_tb = stream_tb(self.fmt, self.audio_idx);
        Ok(())
    }

    /// Mux one packet: shift it to origin-zero, retime it into the output
    /// stream, and hand it to the interleaver (which takes ownership).
    unsafe fn write_packet(
        &mut self,
        packet: *mut ffi::AVPacket,
        out_idx: c_int,
        out_tb: ffi::AVRational,
        src_tb: ffi::AVRational,
        src_start: i64,
    ) -> Result<()> {
        let pkt = unsafe { &mut *packet };
        if pkt.dts == ffi::AV_NOPTS_VALUE {
            pkt.dts = pkt.pts;
        }
        if pkt.pts != ffi::AV_NOPTS_VALUE {
            pkt.pts -= src_start;
        }
        if pkt.dts != ffi::AV_NOPTS_VALUE {
            pkt.dts -= src_start;
        }
        pkt.stream_index = out_idx;
        unsafe { ffi::av_packet_rescale_ts(packet, src_tb, out_tb) };
        check(
            ffi::av_interleaved_write_frame(self.fmt, packet),
            "muxing remuxed packet",
        )
    }

    unsafe fn finish(&mut self) -> Result<()> {
        check(ffi::av_write_trailer(self.fmt), "writing remux trailer")
    }
}

impl Drop for Remuxer {
    fn drop(&mut self) {
        unsafe {
            if !self.fmt.is_null() && !(*self.fmt).pb.is_null() {
                ffi::avio_closep(&mut (*self.fmt).pb);
            }
            if !self.fmt.is_null() {
                ffi::avformat_free_context(self.fmt);
                self.fmt = ptr::null_mut();
            }
        }
    }
}

fn stream_tb(fmt: *mut ffi::AVFormatContext, idx: c_int) -> ffi::AVRational {
    unsafe {
        let stream = *(*fmt).streams.add(idx as usize);
        (*stream).time_base
    }
}

/// The audio side of the merge: either raw source packets or a transcoder
/// standing between the source and the muxer.
enum AudioIn {
    Copy {
        src: *mut ffi::AVFormatContext,
        stream_idx: c_int,
        tb: ffi::AVRational,
        start: i64,
    },
    Transcode(Transcoder),
}

impl AudioIn {
    /// Next audio packet for the muxer (owned by the caller), or `None`
    /// once the audio side is exhausted.
    fn next_packet(&mut self) -> Result<Option<*mut ffi::AVPacket>> {
        match self {
            AudioIn::Copy {
                src, stream_idx, ..
            } => read_next_packet(*src, *stream_idx),
            AudioIn::Transcode(t) => t.next_packet(),
        }
    }

    /// Time base the returned packets' timestamps are expressed in.
    fn tb(&self) -> ffi::AVRational {
        match self {
            AudioIn::Copy { tb, .. } => *tb,
            AudioIn::Transcode(t) => t.enc_tb,
        }
    }

    /// Start time of the audio stream in its own time base; packets are
    /// shifted to start at zero.
    fn start_time(&self) -> i64 {
        match self {
            AudioIn::Copy { start, .. } => *start,
            AudioIn::Transcode(_) => 0,
        }
    }
}

/// Decode → resample → AAC encode chain for one source audio stream.
/// Packets are produced lazily (`next_packet`) so the merge loop can
/// interleave them against the video without buffering the whole track.
///
/// Every pointer field is null unless owned, so Drop doubles as the cleanup
/// path for partially built chains.
struct Transcoder {
    /// Source container; borrowed from the caller's `Input`, which outlives
    /// this transcoder.
    src: *mut ffi::AVFormatContext,
    audio_idx: c_int,
    dec_ctx: *mut ffi::AVCodecContext,
    dec_frame: *mut ffi::AVFrame,
    swr: *mut ffi::SwrContext,
    /// Scratch frame for one resampler batch.
    scratch: *mut ffi::AVFrame,
    /// AAC-frame-sized fltp frame fed to the encoder, filled incrementally.
    staging: *mut ffi::AVFrame,
    enc_ctx: *mut ffi::AVCodecContext,
    /// Encoded packets awaiting pickup by the merge loop.
    queue: VecDeque<*mut ffi::AVPacket>,
    /// Timestamp for the next encoder frame, in samples at AAC_SAMPLE_RATE.
    next_pts: i64,
    /// Valid samples currently in `staging`.
    filled: usize,
    out_channels: usize,
    frame_size: usize,
    enc_tb: ffi::AVRational,
    src_eof: bool,
    enc_eof: bool,
}

// Single-threaded use, like the rest of the remux machinery.
unsafe impl Send for Transcoder {}

impl Transcoder {
    /// Build the chain for `audio_idx` of `src`. Fails cleanly (via Drop on
    /// the partially built value) when a codec or the resampler cannot be
    /// created, e.g. stripped FFmpeg builds without the AAC encoder.
    unsafe fn open(src: *mut ffi::AVFormatContext, audio_idx: c_int) -> Result<Transcoder> {
        let stream = unsafe { *(*src).streams.add(audio_idx as usize) };
        let par = unsafe { (*stream).codecpar };
        let codec_id = unsafe { (*par).codec_id };
        let dec_codec = unsafe { ffi::avcodec_find_decoder(codec_id) };
        if dec_codec.is_null() {
            return Err(format!("no decoder available for {}", codec_name(codec_id)));
        }
        let mut t = Transcoder {
            src,
            audio_idx,
            dec_ctx: ptr::null_mut(),
            dec_frame: ptr::null_mut(),
            swr: ptr::null_mut(),
            scratch: ptr::null_mut(),
            staging: ptr::null_mut(),
            enc_ctx: ptr::null_mut(),
            queue: VecDeque::new(),
            next_pts: 0,
            filled: 0,
            out_channels: 2,
            frame_size: 1024,
            enc_tb: rate_rational(AAC_SAMPLE_RATE),
            src_eof: false,
            enc_eof: false,
        };
        unsafe { t.init(dec_codec, par)? };
        Ok(t)
    }

    unsafe fn init(
        &mut self,
        dec_codec: *const ffi::AVCodec,
        par: *mut ffi::AVCodecParameters,
    ) -> Result<()> {
        self.dec_ctx = unsafe { ffi::avcodec_alloc_context3(dec_codec) };
        if self.dec_ctx.is_null() {
            return Err("allocating audio decoder context: out of memory".into());
        }
        check(
            unsafe { ffi::avcodec_parameters_to_context(self.dec_ctx, par) },
            "configuring audio decoder",
        )?;
        check(
            unsafe { ffi::avcodec_open2(self.dec_ctx, dec_codec, ptr::null_mut()) },
            "opening audio decoder",
        )?;

        let in_rate = unsafe { (*self.dec_ctx).sample_rate };
        let in_channels = unsafe { (*self.dec_ctx).ch_layout.nb_channels };
        if in_rate <= 0 || in_channels <= 0 {
            return Err("audio stream has no usable sample rate or channel layout".into());
        }
        // AAC encodes mono and stereo natively; anything wider is downmixed
        // by the resampler. (Falls back to stereo when no specific layout is
        // known, which the resampler handles.)
        self.out_channels = in_channels.clamp(1, 2) as usize;

        let mut out_layout: ffi::AVChannelLayout = unsafe { std::mem::zeroed() };
        unsafe { ffi::av_channel_layout_default(&mut out_layout, self.out_channels as c_int) };
        check(
            unsafe {
                ffi::swr_alloc_set_opts2(
                    &mut self.swr,
                    &out_layout,
                    ffi::AVSampleFormat::AV_SAMPLE_FMT_FLTP,
                    AAC_SAMPLE_RATE,
                    &(*self.dec_ctx).ch_layout,
                    (*self.dec_ctx).sample_fmt,
                    in_rate,
                    0,
                    ptr::null_mut(),
                )
            },
            "creating audio resampler",
        )?;
        if self.swr.is_null() {
            return Err("creating audio resampler: out of memory".into());
        }
        check(
            unsafe { ffi::swr_init(self.swr) },
            "initializing audio resampler",
        )?;

        let enc_name = CString::new("aac").unwrap();
        let enc_codec = unsafe { ffi::avcodec_find_encoder_by_name(enc_name.as_ptr()) };
        if enc_codec.is_null() {
            return Err("aac encoder not available in this FFmpeg build".into());
        }
        self.enc_ctx = unsafe { ffi::avcodec_alloc_context3(enc_codec) };
        if self.enc_ctx.is_null() {
            return Err("allocating AAC encoder context: out of memory".into());
        }
        unsafe {
            let c = &mut *self.enc_ctx;
            c.sample_rate = AAC_SAMPLE_RATE;
            c.sample_fmt = ffi::AVSampleFormat::AV_SAMPLE_FMT_FLTP;
            ffi::av_channel_layout_default(&mut c.ch_layout, self.out_channels as c_int);
            c.time_base = rate_rational(AAC_SAMPLE_RATE);
            c.bit_rate = if self.out_channels == 1 {
                AAC_BIT_RATE / 2
            } else {
                AAC_BIT_RATE
            };
            // The container keeps the codec config in its header.
            c.flags |= ffi::AV_CODEC_FLAG_GLOBAL_HEADER as c_int;
        }
        check(
            unsafe { ffi::avcodec_open2(self.enc_ctx, enc_codec, ptr::null_mut()) },
            "opening AAC encoder",
        )?;
        self.frame_size = unsafe { (*self.enc_ctx).frame_size.max(1) } as usize;

        self.dec_frame = unsafe { ffi::av_frame_alloc() };
        self.scratch = unsafe { ffi::av_frame_alloc() };
        self.staging = unsafe { ffi::av_frame_alloc() };
        if self.dec_frame.is_null() || self.scratch.is_null() || self.staging.is_null() {
            return Err("allocating audio frames: out of memory".into());
        }
        unsafe {
            let s = &mut *self.staging;
            s.format = ffi::AVSampleFormat::AV_SAMPLE_FMT_FLTP as c_int;
            s.nb_samples = self.frame_size as c_int;
            s.sample_rate = AAC_SAMPLE_RATE;
            ffi::av_channel_layout_default(&mut s.ch_layout, self.out_channels as c_int);
        }
        check(
            unsafe { ffi::av_frame_get_buffer(self.staging, 0) },
            "allocating AAC staging frame",
        )?;
        // The scratch is allocated once here and reused (grown only if a
        // decoder hands back an unusually large frame).
        unsafe { self.ensure_scratch(RESAMPLE_INITIAL_CAPACITY + RESAMPLE_SLACK) }
    }

    /// Next encoded AAC packet, or `None` after the encoder flushed dry.
    /// Pulls and processes as much source input as needed.
    fn next_packet(&mut self) -> Result<Option<*mut ffi::AVPacket>> {
        unsafe { self.next_packet_inner() }
    }

    unsafe fn next_packet_inner(&mut self) -> Result<Option<*mut ffi::AVPacket>> {
        loop {
            if let Some(packet) = self.queue.pop_front() {
                return Ok(Some(packet));
            }
            if self.enc_eof {
                return Ok(None);
            }
            if !self.src_eof {
                match read_next_packet(self.src, self.audio_idx)? {
                    Some(packet) => {
                        self.decode_packet(packet)?;
                        continue;
                    }
                    None => {
                        // End of audio input: flush the decoder, pad and
                        // encode the tail frame, then flush the encoder.
                        self.src_eof = true;
                        let ret = ffi::avcodec_send_packet(self.dec_ctx, ptr::null());
                        if ret < 0 && ret != ffi::AVERROR_EOF {
                            return Err(format!("flushing audio decoder: {}", err_str(ret)));
                        }
                        self.drain_decoder()?;
                        self.encode_tail()?;
                        let ret = ffi::avcodec_send_frame(self.enc_ctx, ptr::null());
                        if ret < 0 && ret != ffi::AVERROR_EOF {
                            return Err(format!("flushing AAC encoder: {}", err_str(ret)));
                        }
                    }
                }
            }
            self.drain_encoder()?;
        }
    }

    /// Send one source packet to the decoder (draining first on EAGAIN),
    /// then process every frame that came out.
    unsafe fn decode_packet(&mut self, packet: *mut ffi::AVPacket) -> Result<()> {
        loop {
            let ret = ffi::avcodec_send_packet(self.dec_ctx, packet);
            if ret == ffi::AVERROR(ffi::EAGAIN) {
                self.drain_decoder()?;
                continue;
            }
            check(ret, "sending audio packet to decoder")?;
            break;
        }
        let mut packet = packet;
        ffi::av_packet_free(&mut packet);
        self.drain_decoder()
    }

    /// Pull decoded frames until the decoder needs more input, feeding each
    /// through the resampler into the staging frame.
    unsafe fn drain_decoder(&mut self) -> Result<()> {
        loop {
            let ret = ffi::avcodec_receive_frame(self.dec_ctx, self.dec_frame);
            if ret == ffi::AVERROR(ffi::EAGAIN) || ret == ffi::AVERROR_EOF {
                return Ok(());
            }
            check(ret, "receiving decoded audio frame")?;
            self.resample_frame()?;
        }
    }

    /// Resample the current decoded frame into `staging`, encoding every
    /// full AAC frame that fills up (one batch can span several).
    unsafe fn resample_frame(&mut self) -> Result<()> {
        let nb_in = (*self.dec_frame).nb_samples;
        if nb_in <= 0 {
            return Ok(());
        }
        // Upper bound on what swr will hand back for this input; fall back
        // to the plain rate conversion if it cannot say.
        let want = unsafe { ffi::swr_get_out_samples(self.swr, nb_in) };
        let cap = if want > 0 {
            want as usize
        } else {
            let in_rate = (*self.dec_ctx).sample_rate.max(1);
            (ffi::av_rescale_q(
                nb_in as i64,
                rate_rational(in_rate),
                rate_rational(AAC_SAMPLE_RATE),
            ) + 8) as usize
        };
        self.ensure_scratch(cap + RESAMPLE_SLACK)?;
        // Both frames below may still be referenced by the encoder (it
        // holds a reference to every frame sent until it releases it), so
        // make them exclusively ours before writing.
        check(
            ffi::av_frame_make_writable(self.scratch),
            "making resample scratch writable",
        )?;
        check(
            ffi::av_frame_make_writable(self.staging),
            "making AAC staging frame writable",
        )?;

        let nb_out = ffi::swr_convert(
            self.swr,
            (*self.scratch).extended_data,
            // Offer the whole scratch, not just `cap`: given only an exact
            // fit swresample switches to an internal-buffering path that
            // (at least on 9.0.2) overruns its own allocations. With room
            // to spare it converts directly and writes exactly `nb_out`.
            (cap + RESAMPLE_SLACK) as c_int,
            (*self.dec_frame).extended_data as *const *const u8,
            nb_in,
        );
        if nb_out < 0 {
            return Err(format!("resampling audio: {}", err_str(nb_out)));
        }

        let mut done = 0usize;
        let total = nb_out.max(0) as usize;
        while done < total {
            let take = (self.frame_size - self.filled).min(total - done);
            for ch in 0..self.out_channels {
                let src = (*self.scratch).data[ch] as *const c_float;
                let dst = (*self.staging).data[ch] as *mut c_float;
                ptr::copy_nonoverlapping(src.add(done), dst.add(self.filled), take);
            }
            self.filled += take;
            done += take;
            if self.filled == self.frame_size {
                self.encode_staging()?;
            }
        }
        Ok(())
    }

    /// Make sure `scratch` owns a buffer that can hold `samples` samples.
    /// The buffer is allocated once and reused; growing unrefs the old one,
    /// which is safe because nothing outside the transcoder keeps a
    /// reference to the scratch between calls.
    unsafe fn ensure_scratch(&mut self, samples: usize) -> Result<()> {
        if !(*self.scratch).buf[0].is_null() && (*self.scratch).nb_samples >= samples as c_int {
            return Ok(());
        }
        ffi::av_frame_unref(self.scratch);
        let s = &mut *self.scratch;
        s.format = ffi::AVSampleFormat::AV_SAMPLE_FMT_FLTP as c_int;
        s.nb_samples = samples as c_int;
        s.sample_rate = AAC_SAMPLE_RATE;
        ffi::av_channel_layout_default(&mut s.ch_layout, self.out_channels as c_int);
        check(
            ffi::av_frame_get_buffer(self.scratch, 0),
            "allocating resample scratch",
        )
    }

    /// Zero-pad the partial staging frame and send it as the final frame
    /// (the AAC encoder only takes exactly `frame_size` samples). The
    /// padding is at most ~21 ms of silence at the very end of the track.
    unsafe fn encode_tail(&mut self) -> Result<()> {
        if self.filled == 0 {
            return Ok(());
        }
        check(
            ffi::av_frame_make_writable(self.staging),
            "making AAC staging frame writable",
        )?;
        for ch in 0..self.out_channels {
            let dst = (*self.staging).data[ch] as *mut c_float;
            // write_bytes counts elements, not bytes.
            ptr::write_bytes(dst.add(self.filled), 0, self.frame_size - self.filled);
        }
        self.encode_staging()
    }

    /// Send the full staging frame to the encoder and collect what it
    /// releases.
    unsafe fn encode_staging(&mut self) -> Result<()> {
        (*self.staging).pts = self.next_pts;
        self.next_pts += self.frame_size as i64;
        loop {
            let ret = ffi::avcodec_send_frame(self.enc_ctx, self.staging);
            if ret == ffi::AVERROR(ffi::EAGAIN) {
                self.drain_encoder()?;
                continue;
            }
            check(ret, "sending audio frame to AAC encoder")?;
            break;
        }
        self.filled = 0;
        self.drain_encoder()
    }

    /// Move every available encoded packet into the queue.
    unsafe fn drain_encoder(&mut self) -> Result<()> {
        loop {
            let mut packet = ffi::av_packet_alloc();
            if packet.is_null() {
                return Err("allocating audio packet: out of memory".into());
            }
            let ret = ffi::avcodec_receive_packet(self.enc_ctx, packet);
            if ret == 0 {
                // Audio has no reordering: a missing dts means pts.
                if (*packet).dts == ffi::AV_NOPTS_VALUE {
                    (*packet).dts = (*packet).pts;
                }
                self.queue.push_back(packet);
            } else if ret == ffi::AVERROR_EOF {
                ffi::av_packet_free(&mut packet);
                self.enc_eof = true;
                return Ok(());
            } else if ret == ffi::AVERROR(ffi::EAGAIN) {
                ffi::av_packet_free(&mut packet);
                return Ok(());
            } else {
                ffi::av_packet_free(&mut packet);
                return Err(format!("receiving AAC packet: {}", err_str(ret)));
            }
        }
    }
}

impl Drop for Transcoder {
    fn drop(&mut self) {
        unsafe {
            while let Some(mut packet) = self.queue.pop_front() {
                ffi::av_packet_free(&mut packet);
            }
            if !self.scratch.is_null() {
                ffi::av_frame_free(&mut self.scratch);
            }
            if !self.staging.is_null() {
                ffi::av_frame_free(&mut self.staging);
            }
            if !self.dec_frame.is_null() {
                ffi::av_frame_free(&mut self.dec_frame);
            }
            if !self.swr.is_null() {
                ffi::swr_free(&mut self.swr);
            }
            if !self.dec_ctx.is_null() {
                ffi::avcodec_free_context(&mut self.dec_ctx);
            }
            if !self.enc_ctx.is_null() {
                ffi::avcodec_free_context(&mut self.enc_ctx);
            }
        }
    }
}

fn rate_rational(rate: c_int) -> ffi::AVRational {
    ffi::AVRational { num: 1, den: rate }
}

/// Timestamp of a packet in seconds, preferring dts (both may be absent on
/// odd streams; 0.0 keeps the merge deterministic then).
unsafe fn packet_seconds(packet: *mut ffi::AVPacket, tb: ffi::AVRational) -> f64 {
    let mut t = (*packet).dts;
    if t == ffi::AV_NOPTS_VALUE {
        t = (*packet).pts;
    }
    if t == ffi::AV_NOPTS_VALUE {
        t = 0;
    }
    t as f64 * tb.num as f64 / tb.den.max(1) as f64
}

/// Name of a codec id for reports ("aac", "opus", ...).
fn codec_name(id: ffi::AVCodecID) -> String {
    unsafe {
        let ptr = ffi::avcodec_get_name(id);
        if ptr.is_null() {
            return "unknown".into();
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Copy `video_out`'s video stream and `source`'s audio track into `tmp`.
fn remux(video_out: &Path, source: &str, mode: Mode, tmp: &Path) -> Result<()> {
    let src = Input::open(Path::new(source))?;
    let vin = Input::open(video_out)?;
    let mut muxer = Remuxer::open(tmp)?;

    unsafe {
        let video_idx = find_video_stream(vin.fmt)?;
        let audio_idx = find_audio_stream(src.fmt)?.ok_or("source has no audio track")?;
        let vstream = *(*vin.fmt).streams.add(video_idx as usize);
        let astream = *(*src.fmt).streams.add(audio_idx as usize);

        muxer.video_idx = muxer.add_stream((*vstream).codecpar)?;

        let mut audio = match mode {
            Mode::Copy => {
                muxer.audio_idx = muxer.add_stream((*astream).codecpar)?;
                AudioIn::Copy {
                    src: src.fmt,
                    stream_idx: audio_idx,
                    tb: (*astream).time_base,
                    start: (*astream).start_time,
                }
            }
            Mode::Transcode => {
                let transcoder = Transcoder::open(src.fmt, audio_idx)?;
                let stream = ffi::avformat_new_stream(muxer.fmt, ptr::null());
                if stream.is_null() {
                    return Err("adding audio stream to remuxer: out of memory".into());
                }
                check(
                    ffi::avcodec_parameters_from_context((*stream).codecpar, transcoder.enc_ctx),
                    "copying AAC codec parameters",
                )?;
                muxer.audio_idx = (*stream).index;
                AudioIn::Transcode(transcoder)
            }
        };

        muxer.write_header()?;

        // Dts-ordered merge: hold at most one packet per side and always
        // hand the interleaver the earlier one, so the file stays tightly
        // interleaved regardless of stream lengths.
        let v_tb = (*vstream).time_base;
        let v_start = (*vstream).start_time;
        let mut pending_video: Option<*mut ffi::AVPacket> = None;
        let mut pending_audio: Option<*mut ffi::AVPacket> = None;
        let mut video_eof = false;
        let mut audio_eof = false;
        loop {
            if pending_video.is_none() && !video_eof {
                match read_next_packet(vin.fmt, video_idx)? {
                    Some(p) => pending_video = Some(p),
                    None => video_eof = true,
                }
            }
            if pending_audio.is_none() && !audio_eof {
                match audio.next_packet()? {
                    Some(p) => pending_audio = Some(p),
                    None => audio_eof = true,
                }
            }
            let (pa, pv) = (pending_audio, pending_video);
            let take_audio = match (pa, pv) {
                (None, None) => break,
                (Some(a), Some(v)) => packet_seconds(a, audio.tb()) < packet_seconds(v, v_tb),
                (Some(_), None) => true,
                (None, Some(_)) => false,
            };
            if take_audio {
                let packet = pending_audio.take().expect("checked above");
                muxer.write_packet(
                    packet,
                    muxer.audio_idx,
                    muxer.audio_tb,
                    audio.tb(),
                    audio.start_time(),
                )?;
            } else {
                let packet = pending_video.take().expect("checked above");
                muxer.write_packet(packet, muxer.video_idx, muxer.video_tb, v_tb, v_start)?;
            }
        }
        muxer.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hwencode;

    fn dir() -> PathBuf {
        std::env::temp_dir().join("anaglyph-audio-tests")
    }

    fn setup() -> PathBuf {
        let dir = dir();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// List `(codec type, codec id)` for every stream in `path`.
    fn probe_streams(path: &Path) -> Vec<(ffi::AVMediaType, ffi::AVCodecID)> {
        let input = Input::open(path).unwrap();
        unsafe {
            (0..(*input.fmt).nb_streams)
                .map(|i| {
                    let stream = *(*input.fmt).streams.add(i as usize);
                    let par = &*(*stream).codecpar;
                    (par.codec_type, par.codec_id)
                })
                .collect()
        }
    }

    /// Write a stereo 48 kHz PCM s16 WAV of a sine tone (a source whose
    /// codec no MP4 muxer will accept).
    fn write_pcm_wav(path: &Path, seconds: i64) {
        unsafe {
            let c_path = CString::new(path.as_os_str().to_string_lossy().as_bytes()).unwrap();
            let mut fmt: *mut ffi::AVFormatContext = ptr::null_mut();
            ffi::avformat_alloc_output_context2(
                &mut fmt,
                ptr::null_mut(),
                ptr::null(),
                c_path.as_ptr(),
            );
            assert!(!fmt.is_null());
            let stream = ffi::avformat_new_stream(fmt, ptr::null());
            let par = &mut *(*stream).codecpar;
            par.codec_type = ffi::AVMediaType::AVMEDIA_TYPE_AUDIO;
            par.codec_id = ffi::AVCodecID::AV_CODEC_ID_PCM_S16LE;
            par.format = ffi::AVSampleFormat::AV_SAMPLE_FMT_S16 as c_int;
            par.sample_rate = 48_000;
            ffi::av_channel_layout_default(&mut par.ch_layout, 2);
            assert!(
                ffi::avio_open2(
                    &mut (*fmt).pb,
                    c_path.as_ptr(),
                    ffi::AVIO_FLAG_WRITE,
                    ptr::null(),
                    ptr::null_mut()
                ) >= 0
            );
            assert!(ffi::avformat_write_header(fmt, ptr::null_mut()) >= 0);

            let nb = 48_000 * seconds;
            let packet = ffi::av_packet_alloc();
            assert!(ffi::av_new_packet(packet, (nb * 4) as c_int) >= 0);
            let samples =
                std::slice::from_raw_parts_mut((*packet).data as *mut i16, (nb * 2) as usize);
            for (i, sample) in samples.chunks_mut(2).enumerate() {
                let value = ((i as f64 * 2.0 * std::f64::consts::TAU * 440.0 / 48_000.0).sin()
                    * 12_000.0) as i16;
                sample[0] = value;
                sample[1] = value;
            }
            (*packet).pts = 0;
            (*packet).dts = 0;
            (*packet).duration = nb;
            assert!(ffi::av_interleaved_write_frame(fmt, packet) >= 0);
            assert!(ffi::av_write_trailer(fmt) >= 0);
            ffi::avio_closep(&mut (*fmt).pb);
            ffi::avformat_free_context(fmt);
        }
    }

    #[test]
    fn silent_source_reports_none_and_touches_nothing() {
        if !hwencode::software_encoder_available() {
            return;
        }
        let dir = setup();
        let video = dir.join("silent.mp4");
        let _ = std::fs::remove_file(&video);
        hwencode::make_test_video(&video, 4).unwrap();
        let before = std::fs::read(&video).unwrap();

        let desc = attach_audio(video.to_str().unwrap(), &video).unwrap();

        assert!(desc.contains("no audio track"), "desc: {desc}");
        assert_eq!(std::fs::read(&video).unwrap(), before);
    }

    #[test]
    fn mp4_compatible_audio_is_stream_copied() {
        // booth2.mp4 carries AAC audio, which MP4 accepts natively.
        let booth = Path::new(env!("CARGO_MANIFEST_DIR")).join("test/booth2.mp4");
        if !booth.exists() || !hwencode::software_encoder_available() {
            return;
        }
        let dir = setup();
        let target = dir.join("copy-target.mp4");
        let _ = std::fs::remove_file(&target);
        hwencode::make_test_video(&target, 4).unwrap();

        let desc = attach_audio(booth.to_str().unwrap(), &target).unwrap();
        assert!(desc.contains("stream copy"), "desc: {desc}");

        let streams = probe_streams(&target);
        assert_eq!(streams.len(), 2, "expected video + audio: {streams:?}");
        assert_eq!(
            streams[1].1,
            ffi::AVCodecID::AV_CODEC_ID_AAC,
            "copied stream must keep its codec"
        );
    }

    #[test]
    fn unsupported_audio_is_transcoded_to_aac() {
        let dir = setup();
        let wav = dir.join("tone.wav");
        let target = dir.join("transcode-target.mp4");
        let _ = std::fs::remove_file(&wav);
        let _ = std::fs::remove_file(&target);
        write_pcm_wav(&wav, 1);
        if !hwencode::software_encoder_available() {
            return;
        }
        hwencode::make_test_video(&target, 4).unwrap();

        let desc = attach_audio(wav.to_str().unwrap(), &target).unwrap();
        assert!(desc.contains("transcoded from pcm_s16le"), "desc: {desc}");

        let streams = probe_streams(&target);
        assert_eq!(streams.len(), 2, "expected video + audio: {streams:?}");
        assert_eq!(streams[1].1, ffi::AVCodecID::AV_CODEC_ID_AAC);
    }

    #[test]
    fn full_conversion_carries_audio() {
        let booth = Path::new(env!("CARGO_MANIFEST_DIR")).join("test/booth2.mp4");
        if !booth.exists() {
            return;
        }
        let dir = setup();
        let out = dir.join("converted.mp4");
        let _ = std::fs::remove_file(&out);

        let report = crate::video::convert_video_to_anaglyph(
            booth.to_str().unwrap(),
            out.to_str().unwrap(),
            crate::video::DirectionChoice::Fixed(crate::anaglyph::VideoDirection::Clockwise),
            crate::anaglyph::AnaglyphType::True,
            crate::video::DecoderChoice::Cpu,
            crate::video::EncoderChoice::Software,
            crate::anaglyph::BlenderChoice::Cpu,
        )
        .unwrap();

        assert!(
            report.audio.contains("stream copy"),
            "audio: {}",
            report.audio
        );
        let streams = probe_streams(&out);
        assert_eq!(streams.len(), 2, "expected video + audio: {streams:?}");
    }
}
