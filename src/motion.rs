//! Frame-to-frame motion analysis for temporal-stereo ("360 booth") videos.
//!
//! A booth camera rides an arc around the subject: consecutive frames form a
//! stereo pair, and which frame feeds the *left* eye depends on the direction
//! the camera travels along the arc. When the booth reverses for a
//! "boomerang" pass the pairing has to swap, and the reversal can be read off
//! the images themselves. The camera rotates to keep the subject framed, so
//! that tracking rotation dominates the image motion and the background
//! slides in the *same* direction the camera moves: content drifting right
//! means the camera moved right, i.e. the newer frame is the right-eye view.
//!
//! [`MotionEstimator`] measures that per-frame slide; [`DirectionTracker`]
//! integrates the slide into a zig-zag trend, reports confirmed reversal
//! points, and tracks how far a streaming pipeline may emit frames without
//! guessing. Pure CPU, no dependencies, usable from wasm.

use crate::anaglyph::VideoDirection;

/// Measured horizontal motion of the image content between two frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionSample {
    /// Signed horizontal drift in source pixels per frame, positive when the
    /// image content moved right (the camera moved left).
    pub dx: f32,
    /// `false` when the frames could not be matched reliably: scene cut,
    /// motion beyond the search range, or textureless input.
    pub reliable: bool,
}

/// Width the frames are shrunk to before block matching; the motion estimate
/// does not need detail, and small frames keep the per-frame cost far below
/// one video frame interval.
const TARGET_WIDTH: usize = 256;
/// Side of one block-matching block, in matched (downsampled) pixels.
const BLOCK: usize = 8;
/// Search radius for the horizontal shift, in matched pixels.
const MAX_SHIFT: usize = 12;
/// Blocks whose gray-level variance is below this carry no information to
/// match on (featureless walls) and are skipped.
const MIN_BLOCK_VARIANCE: f32 = 6.0;
/// A best-match SAD above this fraction of the block deviation means the
/// frames simply do not show the same scene (a cut, or untrackable motion).
const CUT_QUALITY_LIMIT: f32 = 0.45;
/// Fewer than this fraction of the block grid matchable -> unreliable sample.
const MIN_BLOCK_RATIO: f32 = 0.2;

/// Estimates the horizontal drift between consecutive frames of a video.
///
/// Frames are box-downsampled to grayscale and matched block-by-block; the
/// per-block shifts are reduced with a median so a subject moving against the
/// camera (or a static subject, where those blocks are flat) cannot sway the
/// result, which is dominated by the background the camera orbits.
pub struct MotionEstimator {
    src_width: usize,
    width: usize,
    height: usize,
    /// Matched-pixel to source-pixel scale (`src = matched / scale`).
    scale: f32,
    x_ranges: Vec<(u32, u32)>,
    y_ranges: Vec<(u32, u32)>,
    prev: Vec<u8>,
    /// Scratch for the current frame's grayscale; swapped with `prev` after
    /// matching so steady-state observation allocates nothing.
    cur: Vec<u8>,
    have_prev: bool,
}

impl MotionEstimator {
    /// New estimator for frames of `width` x `height` RGB24 pixels.
    pub fn new(width: usize, height: usize) -> Self {
        let (sw, sh, scale) = if width > TARGET_WIDTH {
            let scale = TARGET_WIDTH as f32 / width as f32;
            (
                TARGET_WIDTH,
                ((height as f32 * scale) as usize).max(1),
                scale,
            )
        } else {
            (width, height, 1.0)
        };
        let ranges = |small: usize, big: usize| -> Vec<(u32, u32)> {
            (0..small)
                .map(|i| {
                    let start = i * big / small;
                    let end = ((i + 1) * big / small).max(start + 1);
                    (start as u32, end as u32)
                })
                .collect()
        };
        MotionEstimator {
            src_width: width,
            width: sw,
            height: sh,
            scale,
            x_ranges: ranges(sw, width),
            y_ranges: ranges(sh, height),
            prev: Vec::new(),
            cur: Vec::new(),
            have_prev: false,
        }
    }

    /// Match the next frame (`rgb`, dense RGB24 of the configured size)
    /// against its predecessor and advance the internal state.
    pub fn observe(&mut self, rgb: &[u8]) -> MotionSample {
        let mut cur = std::mem::take(&mut self.cur);
        cur.resize(self.width * self.height, 0);
        self.downsample(&mut cur, rgb);
        self.advance(cur)
    }

    /// Match the next frame (dense RGBA of the configured size) against its
    /// predecessor and advance the internal state. The box-downsample reads
    /// the same R/G/B values through the same BT.601 weights as the RGB24
    /// [`Self::observe`], so both entry points produce identical samples;
    /// accepting RGBA directly skips a whole packing pass on the caller
    /// side (the web pipeline blends and copies RGBA throughout).
    pub fn observe_rgba(&mut self, rgba: &[u8]) -> MotionSample {
        let mut cur = std::mem::take(&mut self.cur);
        cur.resize(self.width * self.height, 0);
        self.downsample_rgba(&mut cur, rgba);
        self.advance(cur)
    }

    /// Reduce the freshly downsampled `cur` against the stored predecessor,
    /// then make `cur` the new predecessor (the old one becomes next
    /// frame's scratch).
    fn advance(&mut self, cur: Vec<u8>) -> MotionSample {
        let sample = if self.have_prev {
            self.match_frames(&cur)
        } else {
            // Nothing to compare the first frame against.
            MotionSample {
                dx: 0.0,
                reliable: false,
            }
        };
        self.cur = std::mem::replace(&mut self.prev, cur);
        self.have_prev = true;
        sample
    }

    /// Box-downsample RGB24 to the matched-size grayscale buffer (`out`
    /// holds `width * height` pixels).
    fn downsample(&self, out: &mut [u8], rgb: &[u8]) {
        for (oy, &(y0, y1)) in self.y_ranges.iter().enumerate() {
            for (ox, &(x0, x1)) in self.x_ranges.iter().enumerate() {
                let mut sum = 0u32;
                let mut n = 0u32;
                for y in y0..y1 {
                    let row = y as usize * self.src_width;
                    let start = (row + x0 as usize) * 3;
                    let end = (row + x1 as usize) * 3;
                    for px in rgb[start..end].as_chunks::<3>().0 {
                        // BT.601 luma with integer weights summing to 256.
                        sum += (77 * u32::from(px[0])
                            + 150 * u32::from(px[1])
                            + 29 * u32::from(px[2]))
                            >> 8;
                        n += 1;
                    }
                }
                out[oy * self.width + ox] = (sum / n.max(1)) as u8;
            }
        }
    }

    /// Box-downsample RGBA to the matched-size grayscale buffer. Identical
    /// pixel ranges and luma weights as [`Self::downsample`]; alpha is
    /// ignored (decoder output is opaque).
    fn downsample_rgba(&self, out: &mut [u8], rgba: &[u8]) {
        for (oy, &(y0, y1)) in self.y_ranges.iter().enumerate() {
            for (ox, &(x0, x1)) in self.x_ranges.iter().enumerate() {
                let mut sum = 0u32;
                let mut n = 0u32;
                for y in y0..y1 {
                    let row = y as usize * self.src_width;
                    let start = (row + x0 as usize) * 4;
                    let end = (row + x1 as usize) * 4;
                    for px in rgba[start..end].as_chunks::<4>().0 {
                        // BT.601 luma with integer weights summing to 256.
                        sum += (77 * u32::from(px[0])
                            + 150 * u32::from(px[1])
                            + 29 * u32::from(px[2]))
                            >> 8;
                        n += 1;
                    }
                }
                out[oy * self.width + ox] = (sum / n.max(1)) as u8;
            }
        }
    }

    /// Median block shift between the stored previous frame and `cur`.
    fn match_frames(&self, cur: &[u8]) -> MotionSample {
        let unreliable = || MotionSample {
            dx: 0.0,
            reliable: false,
        };
        // Blocks need `MAX_SHIFT` of margin on both sides to search in.
        if self.width < 2 * MAX_SHIFT + BLOCK || self.height < BLOCK {
            return unreliable();
        }
        let mut shifts: Vec<f32> = Vec::new();
        let mut grid = 0usize;
        let mut sad_sum = 0.0f32;
        let mut dev_sum = 0.0f32;
        for by in (0..=self.height - BLOCK).step_by(BLOCK) {
            let mut bx = MAX_SHIFT;
            while bx + BLOCK + MAX_SHIFT <= self.width {
                grid += 1;
                if let Some((shift, sad, dev)) = self.match_block(by, bx, cur) {
                    shifts.push(shift);
                    sad_sum += sad;
                    dev_sum += dev;
                }
                bx += BLOCK;
            }
        }
        let qualified = shifts.len();
        if qualified == 0 || (qualified as f32) < grid as f32 * MIN_BLOCK_RATIO {
            return unreliable();
        }
        // Mean best-match error relative to the mean block deviation: near 0
        // for a solid match, far above the cut limit when the frames show
        // unrelated content.
        let quality = (sad_sum / qualified as f32) / (dev_sum / qualified as f32 + 1.0);
        if quality > CUT_QUALITY_LIMIT {
            return unreliable();
        }
        shifts.sort_unstable_by(f32::total_cmp);
        MotionSample {
            dx: shifts[qualified / 2] / self.scale,
            reliable: true,
        }
    }

    /// Best horizontal shift for one block as `(subpixel shift in matched
    /// pixels, mean SAD, block deviation)`, or `None` when the block is too
    /// flat to match or its best shift saturated the search range.
    fn match_block(&self, by: usize, bx: usize, cur: &[u8]) -> Option<(f32, f32, f32)> {
        let prev = &self.prev;
        let w = self.width;
        let mut sum = 0i64;
        let mut sum_sq = 0i64;
        for y in by..by + BLOCK {
            for x in bx..bx + BLOCK {
                let v = i64::from(prev[y * w + x]);
                sum += v;
                sum_sq += v * v;
            }
        }
        let n = (BLOCK * BLOCK) as f32;
        let mean = sum as f32 / n;
        let variance = (sum_sq as f32 / n - mean * mean).max(0.0);
        if variance < MIN_BLOCK_VARIANCE {
            return None;
        }

        let mut best_d = 0i32;
        let mut best_sad = u64::MAX;
        for d in -(MAX_SHIFT as i32)..=(MAX_SHIFT as i32) {
            let sad = Self::block_sad(prev, cur, w, bx, by, d);
            if sad < best_sad {
                best_sad = sad;
                best_d = d;
            }
        }
        if best_d.abs() == MAX_SHIFT as i32 {
            // The true shift is outside the search window; this block knows
            // nothing usable (and its SAD curve cannot judge match quality).
            return None;
        }
        // Parabolic subpixel refinement from the SAD curve around the best
        // integer shift.
        let mut shift = best_d as f32;
        let sm = Self::block_sad(prev, cur, w, bx, by, best_d - 1) as i64;
        let sp = Self::block_sad(prev, cur, w, bx, by, best_d + 1) as i64;
        let denom = sm + sp - 2 * best_sad as i64;
        if denom > 0 {
            let offset = (sm - sp) as f32 / (2 * denom) as f32;
            if offset.is_finite() {
                shift += offset.clamp(-0.5, 0.5);
            }
        }
        Some((shift, best_sad as f32 / n, variance.sqrt()))
    }

    /// Sum of absolute differences between the block at `(bx, by)` in `prev`
    /// and the same block sampled `d` pixels further right in `cur`:
    /// `d > 0` means the content moved right.
    fn block_sad(prev: &[u8], cur: &[u8], w: usize, bx: usize, by: usize, d: i32) -> u64 {
        let shift = d.unsigned_abs() as usize;
        let mut acc = 0u64;
        for y in by..by + BLOCK {
            let row = y * w;
            let p = &prev[row + bx..row + bx + BLOCK];
            let c = if d >= 0 {
                &cur[row + bx + shift..row + bx + shift + BLOCK]
            } else {
                &cur[row + bx - shift..row + bx - shift + BLOCK]
            };
            acc += u64::from(
                p.iter()
                    .zip(c)
                    .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
                    .sum::<u32>(),
            );
        }
        acc
    }
}

/// Direction of the integrated image drift over the current segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    /// Image content drifting right (camera moving left).
    Increasing,
    /// Image content drifting left (camera moving right).
    Decreasing,
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
pub fn trend_to_direction(trend: Option<Trend>, reverse: bool) -> VideoDirection {
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

/// A confirmed reversal of the drift trend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reversal {
    /// Index of the first frame belonging to the new trend.
    pub frame: usize,
    /// Source timestamp of the turning point, in seconds.
    pub time: f64,
    pub from: Trend,
    pub to: Trend,
}

/// Tuning knobs for [`DirectionTracker`]; drift units are source pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackerConfig {
    /// Drift that must accumulate before the first direction is declared,
    /// expressed as frames of motion at the observed speed (so a slow drift
    /// waits longer, like the reversal threshold below).
    pub start_frames: f32,
    /// Counter-drift needed to confirm a reversal, in frames of motion at
    /// the observed speed: the integrated retreat must reach
    /// `swing_frames * speed` before the turning point is called.
    pub swing_frames: f32,
    /// Floor for both thresholds in source pixels, so near-stationary
    /// sections need more than noise to flip the trend.
    pub min_drift: f32,
    /// Frames of ambiguity tolerated before a stretch is force-committed to
    /// the current trend; bounds output latency and frame buffering. Only
    /// reachable when the motion stalls at a turning point (drift too small
    /// to accumulate a swing), where the pairing hardly matters anyway.
    pub pending_cap: usize,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        TrackerConfig {
            start_frames: 4.0,
            swing_frames: 8.0,
            min_drift: 4.0,
            pending_cap: 20,
        }
    }
}

/// Turns a stream of [`MotionSample`]s into trend segments and reversal
/// points, and tracks how far downstream frames may be committed.
///
/// The integrated drift is a zig-zag whose turning points are the reversals,
/// but a turning point is only provable once the drift has retreated from it
/// by the hysteresis threshold — `swing_frames` frames of motion at the
/// observed speed (so the wait is proportional to how fast the camera
/// actually moves, and immune to noise and to the subject's own motion). The
/// tracker therefore reports a `confirmed_through` frame index: every frame
/// at or below it is guaranteed to keep its assigned trend. This never stalls
/// a steady run for longer than the hysteresis needs, because a future
/// reversal can only happen at a future drift extremum: frames up to the
/// *current* extremum candidate are already final even while the trend
/// itself might still flip.
#[derive(Debug)]
pub struct DirectionTracker {
    cfg: TrackerConfig,
    /// Integrated drift over accepted samples.
    drift: f32,
    /// Exponential moving average of |dx|: the observed motion speed, used
    /// to scale the hysteresis thresholds.
    speed: f32,
    trend: Option<Trend>,
    /// Value, frame and timestamp of the current trend's extremum candidate:
    /// the point where the next reversal, if any, will be placed.
    ext_value: f32,
    ext_frame: usize,
    ext_time: f64,
    last_frame: Option<usize>,
    confirmed: Option<usize>,
    reversals: Vec<Reversal>,
}

/// Smoothing factor of the speed estimate: responsive enough to follow the
/// booth speeding up and easing out, smooth enough to ignore single-frame
/// measurement spikes.
const SPEED_SMOOTHING: f32 = 0.1;

impl DirectionTracker {
    pub fn new(cfg: TrackerConfig) -> Self {
        DirectionTracker {
            cfg,
            drift: 0.0,
            speed: 0.0,
            trend: None,
            ext_value: 0.0,
            ext_frame: 0,
            ext_time: 0.0,
            last_frame: None,
            confirmed: None,
            reversals: Vec::new(),
        }
    }

    /// Record the sample that *ends* at `frame` (the motion from `frame - 1`
    /// to `frame`) at source time `time` seconds. Frames must arrive in
    /// increasing order.
    pub fn push(&mut self, frame: usize, time: f64, sample: &MotionSample) {
        self.last_frame = Some(frame);
        if !sample.reliable {
            // A cut (or an unmatchable pair) says nothing about the trend,
            // but holding frames back for it buys nothing either: commit the
            // pending stretch and carry the current trend across the cut.
            self.ext_frame = frame;
            self.ext_time = time;
            self.confirmed = Some(frame);
            return;
        }
        self.drift += sample.dx;
        self.speed = if self.speed == 0.0 {
            sample.dx.abs()
        } else {
            self.speed * (1.0 - SPEED_SMOOTHING) + sample.dx.abs() * SPEED_SMOOTHING
        };
        let Some(trend) = self.trend else {
            // No direction yet: wait for enough drift to accumulate to tell
            // a slow start from noise.
            let start = (self.cfg.start_frames * self.speed).max(self.cfg.min_drift);
            if self.drift >= start {
                self.trend = Some(Trend::Increasing);
            } else if self.drift <= -start {
                self.trend = Some(Trend::Decreasing);
            } else {
                return;
            }
            self.ext_value = self.drift;
            self.ext_frame = frame;
            self.ext_time = time;
            self.confirmed = Some(frame);
            return;
        };
        let made_new_extremum = match trend {
            Trend::Increasing => self.drift > self.ext_value,
            Trend::Decreasing => self.drift < self.ext_value,
        };
        if made_new_extremum {
            self.ext_value = self.drift;
            self.ext_frame = frame;
            self.ext_time = time;
        } else if (self.drift - self.ext_value).abs()
            >= (self.cfg.swing_frames * self.speed).max(self.cfg.min_drift)
        {
            let to = match trend {
                Trend::Increasing => Trend::Decreasing,
                Trend::Decreasing => Trend::Increasing,
            };
            self.reversals.push(Reversal {
                frame: self.ext_frame + 1,
                time: self.ext_time,
                from: trend,
                to,
            });
            // The current frame is the starting extremum candidate of the new
            // trend: the drift has been moving away from the turning point
            // ever since, so it is the new extreme.
            self.trend = Some(to);
            self.ext_value = self.drift;
            self.ext_frame = frame;
            self.ext_time = time;
        } else if frame - self.ext_frame > self.cfg.pending_cap {
            // A long ambiguous stretch means near-zero motion, where the
            // pairing hardly matters: commit it to the current trend to bound
            // latency. The extremum *value* is kept, so a later reversal is
            // still measured against the true turning point.
            self.ext_frame = frame;
            self.ext_time = time;
        }
        self.confirmed = Some(
            self.confirmed
                .map_or(self.ext_frame, |c| c.max(self.ext_frame)),
        );
    }

    /// End of stream: everything held back is now final, on the current
    /// trend (or the caller's default if no trend ever formed).
    pub fn finish(&mut self) {
        if let Some(frame) = self.last_frame {
            self.ext_frame = frame;
            self.confirmed = Some(frame);
        }
    }

    /// Highest frame index whose trend assignment can no longer change.
    pub fn confirmed_through(&self) -> Option<usize> {
        self.confirmed
    }

    /// Trend of the frames following `confirmed_through()`, once decided.
    pub fn current_trend(&self) -> Option<Trend> {
        self.trend
    }

    /// Drain the reversals confirmed since the last call.
    pub fn take_reversals(&mut self) -> Vec<Reversal> {
        std::mem::take(&mut self.reversals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(dx: f32) -> MotionSample {
        MotionSample { dx, reliable: true }
    }

    fn feed(tracker: &mut DirectionTracker, start: usize, dxs: &[f32]) {
        for (i, dx) in dxs.iter().enumerate() {
            let frame = start + i;
            tracker.push(frame, frame as f64 / 30.0, &sample(*dx));
        }
    }

    #[test]
    fn zigzag_finds_turning_points() {
        let mut t = DirectionTracker::new(TrackerConfig::default());
        feed(&mut t, 1, &[2.0; 30]); // rightward through frame 30
        assert_eq!(t.current_trend(), Some(Trend::Increasing));
        assert_eq!(t.confirmed_through(), Some(30));
        feed(&mut t, 31, &[-2.0; 30]); // leftward through frame 60
        let r = t.take_reversals();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].frame, 31);
        assert_eq!((r[0].from, r[0].to), (Trend::Increasing, Trend::Decreasing));
        feed(&mut t, 61, &[2.0; 30]);
        let r = t.take_reversals();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].frame, 61);
        assert_eq!((r[0].from, r[0].to), (Trend::Decreasing, Trend::Increasing));
        assert_eq!(t.current_trend(), Some(Trend::Increasing));
    }

    #[test]
    fn noisy_drift_still_detects_both_reversals() {
        // Deterministic pseudo-noise on top of the drift.
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut noise = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 1.4 - 0.7
        };
        let mut t = DirectionTracker::new(TrackerConfig::default());
        feed(
            &mut t,
            1,
            &(0..40).map(|_| 2.0 + noise()).collect::<Vec<_>>(),
        );
        feed(
            &mut t,
            41,
            &(0..40).map(|_| -2.0 + noise()).collect::<Vec<_>>(),
        );
        feed(
            &mut t,
            81,
            &(0..40).map(|_| 2.0 + noise()).collect::<Vec<_>>(),
        );
        let r = t.take_reversals();
        assert_eq!(r.len(), 2, "reversals: {r:?}");
        assert!(
            r[0].frame.abs_diff(41) <= 3 && r[1].frame.abs_diff(81) <= 3,
            "reversals: {r:?}"
        );
        assert_eq!(t.current_trend(), Some(Trend::Increasing));
    }

    #[test]
    fn holds_frames_until_the_trend_is_decided() {
        let mut t = DirectionTracker::new(TrackerConfig::default());
        feed(&mut t, 1, &[0.5; 7]); // drift 3.5 < start threshold 4
        assert_eq!(t.current_trend(), None);
        assert_eq!(t.confirmed_through(), None);
        feed(&mut t, 8, &[0.5; 1]); // drift 4.0 decides it
        assert_eq!(t.current_trend(), Some(Trend::Increasing));
        assert_eq!(t.confirmed_through(), Some(8));
    }

    #[test]
    fn scene_cut_commits_pending_but_is_not_a_reversal() {
        let mut t = DirectionTracker::new(TrackerConfig::default());
        feed(&mut t, 1, &[2.0; 19]);
        t.push(
            20,
            20.0 / 30.0,
            &MotionSample {
                dx: 0.0,
                reliable: false,
            },
        );
        assert_eq!(t.confirmed_through(), Some(20));
        assert_eq!(t.current_trend(), Some(Trend::Increasing));
        feed(&mut t, 21, &[2.0; 10]); // drift keeps rising through the cut
        assert!(t.take_reversals().is_empty());
        feed(&mut t, 31, &[-2.0; 30]);
        let r = t.take_reversals();
        assert_eq!(r.len(), 1);
        // The true extremum (frame 30, before the drift turned) is preserved
        // across the forced commit.
        assert_eq!(r[0].frame, 31);
    }

    #[test]
    fn pending_cap_bounds_ambiguity() {
        let mut t = DirectionTracker::new(TrackerConfig::default());
        feed(&mut t, 1, &[2.0; 25]); // Increasing decided at frame 4, ext at 25
        feed(&mut t, 26, &[0.0; 25]); // flat: no new extremum, no retreat
                                      // The cap force-commits the ambiguous stretch ~20 frames after the
                                      // extremum instead of holding it forever.
        assert!(t.confirmed_through().unwrap() >= 45);
        assert!(t.take_reversals().is_empty());
        feed(&mut t, 51, &[-2.0; 30]);
        let r = t.take_reversals();
        assert_eq!(r.len(), 1);
        // The boundary moved to just past the commit point (not the extremum
        // at frame 25), but the swing was still measured against the extremum.
        assert!((45..=47).contains(&r[0].frame), "reversal: {r:?}",);
    }

    #[test]
    fn finish_commits_the_tail_without_a_reversal() {
        let mut t = DirectionTracker::new(TrackerConfig::default());
        feed(&mut t, 1, &[2.0; 20]);
        feed(&mut t, 21, &[-0.5; 10]); // retreat 5 < swing: not a reversal
        assert_eq!(t.current_trend(), Some(Trend::Increasing));
        assert!(t.confirmed_through().unwrap() < 30);
        t.finish();
        assert_eq!(t.confirmed_through(), Some(30));
        assert!(t.take_reversals().is_empty());
    }

    #[test]
    fn finish_without_ever_deciding_a_trend() {
        let mut t = DirectionTracker::new(TrackerConfig::default());
        feed(&mut t, 1, &[0.1; 10]);
        assert_eq!(t.confirmed_through(), None);
        t.finish();
        assert_eq!(t.confirmed_through(), Some(10));
        assert_eq!(t.current_trend(), None);
    }

    // --- estimator tests ---------------------------------------------------

    /// Pseudo-random grayscale master the synthetic frames are cropped from.
    fn master(w: usize, h: usize, seed: u64) -> Vec<u8> {
        let mut v = vec![0u8; w * h];
        let mut state = seed;
        for p in v.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *p = (state >> 33) as u8;
        }
        v
    }

    /// Box-blurred copy of the master, standing in for the spatial
    /// correlation of real footage: raw per-pixel noise decorrelates under
    /// the estimator's downsampling, which no actual camera does.
    fn blur(master: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0u32;
                let mut n = 0u32;
                for sy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                    for sx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                        sum += u32::from(master[sy * w + sx]);
                        n += 1;
                    }
                }
                out[y * w + x] = (sum / n) as u8;
            }
        }
        out
    }

    /// Crop a `w x h` window at x offset `off` from a grayscale master and
    /// expand it to dense RGB24.
    fn crop_rgb(master: &[u8], mw: usize, off: usize, w: usize, h: usize) -> Vec<u8> {
        let mut rgb = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let g = master[y * mw + off + x];
                rgb.extend_from_slice(&[g, g, g]);
            }
        }
        rgb
    }

    #[test]
    fn estimator_measures_a_rightward_shift() {
        let (w, h) = (200, 120);
        let m = master(w + 20, h, 7);
        // A feature at master column c sits at x = c-5 in `prev` and x = c in
        // `cur`: the content moved 5 px right.
        let prev = crop_rgb(&m, w + 20, 5, w, h);
        let cur = crop_rgb(&m, w + 20, 0, w, h);
        let mut est = MotionEstimator::new(w, h);
        est.observe(&prev);
        let s = est.observe(&cur);
        assert!(s.reliable, "textured pair should match");
        assert!((s.dx - 5.0).abs() <= 1.0, "dx = {}", s.dx);
    }

    #[test]
    fn observe_rgba_matches_observe_bit_for_bit() {
        // The RGBA fast path (web pipeline) must produce exactly the same
        // samples as the RGB24 path for the same R/G/B values, whatever the
        // alpha lane holds.
        let (w, h) = (200, 120);
        let m = master(w + 20, h, 7);
        let mut rgb_est = MotionEstimator::new(w, h);
        let mut rgba_est = MotionEstimator::new(w, h);
        for step in 0..6usize {
            let rgb = crop_rgb(&m, w + 20, step * 3, w, h);
            let mut rgba = Vec::with_capacity(w * h * 4);
            for (i, px) in rgb.as_chunks::<3>().0.iter().enumerate() {
                // Reordered channels + garbage alpha: identical per-pixel
                // R/G/B, so both paths must agree bit for bit.
                rgba.extend_from_slice(&[px[2], px[0], px[1], (i * 37) as u8]);
            }
            let a = rgb_est.observe(&rgb);
            let b = rgba_est.observe_rgba(&rgba);
            assert_eq!(a, b, "step {step}");
        }
    }

    #[test]
    fn estimator_measures_a_leftward_shift_when_downsampled() {
        let (w, h) = (1000, 400);
        let m = blur(&master(w + 20, h, 11), w + 20, h, 3);
        // Mirror image of the test above: content moves 10 px left, and the
        // estimator runs through its downsampled path (scale = 256/1000).
        let prev = crop_rgb(&m, w + 20, 0, w, h);
        let cur = crop_rgb(&m, w + 20, 10, w, h);
        let mut est = MotionEstimator::new(w, h);
        est.observe(&prev);
        let s = est.observe(&cur);
        assert!(s.reliable);
        assert!((s.dx + 10.0).abs() <= 3.0, "dx = {}", s.dx);
    }

    #[test]
    fn median_resists_a_subject_moving_the_wrong_way() {
        let (w, h) = (240, 120);
        let m = blur(&master(w + 10, h, 5), w + 10, h, 2);
        let frame = |off: usize, blob_x: usize| -> Vec<u8> {
            let mut rgb = crop_rgb(&m, w + 10, off, w, h);
            for y in 20..100usize {
                for x in blob_x..blob_x + 40 {
                    let i = (y * w + x) * 3;
                    rgb[i] = 0;
                    rgb[i + 1] = 0;
                    rgb[i + 2] = 0;
                }
            }
            rgb
        };
        let mut est = MotionEstimator::new(w, h);
        est.observe(&frame(6, 60));
        // Background slides 4 px right while the "subject" slides 40 px left.
        let s = est.observe(&frame(2, 20));
        assert!(s.reliable);
        assert!((s.dx - 4.0).abs() <= 1.5, "dx = {}", s.dx);
    }

    #[test]
    fn estimator_flags_unrelated_frames() {
        let (w, h) = (200, 120);
        let a = crop_rgb(&master(w, h, 3), w, 0, w, h);
        let b = crop_rgb(&master(w, h, 99), w, 0, w, h);
        let mut est = MotionEstimator::new(w, h);
        assert!(!est.observe(&a).reliable, "first frame has no match");
        assert!(!est.observe(&b).reliable, "unrelated frames are a cut");
        assert!(est.observe(&b).reliable, "a repeated frame matches itself");
    }

    #[test]
    fn estimator_flags_textureless_frames() {
        let (w, h) = (64, 48);
        let flat = vec![128u8; w * h * 3];
        let mut est = MotionEstimator::new(w, h);
        est.observe(&flat);
        assert!(!est.observe(&flat).reliable);
    }
}
