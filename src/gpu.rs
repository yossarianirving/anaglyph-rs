//! GPU-accelerated anaglyph frame blending using a wgpu compute shader.
//!
//! Combining two RGB24 frames into an anaglyph is a per-pixel linear
//! transform: `out[c] = dot(M_left[c], left.rgb) + dot(M_right[c], right.rgb)`.
//! This is embarrassingly parallel, so we run it on the GPU instead of a
//! scalar loop on the CPU.
//!
//! Frames are exchanged as dense byte buffers: RGB24 (`3 * width * height`,
//! video) or RGBA (`4 * width * height`, gifs). On the GPU, RGB24 input
//! pixels are unpacked from the byte stream directly (each pixel may span
//! two 32-bit words); RGBA pixels are one word each. Output pixels are always
//! written as packed `0x00BBGGRR` words, which map to a byte order of
//! `R, G, B, 0` in memory, so the RGB24 readback only needs a
//! stride 4 -> 3 unpad.
//!
//! Sums are rounded (half away from zero) and clamped to `0..=255` before
//! quantizing, matching the CPU reference in `anaglyph::combine_slices`.
//!
//! The synchronous API (`new`, `blend`, `blend_rgba`) is native-only: it
//! waits on the GPU with a blocking poll and returns an error (rather than
//! panicking) when the device is lost or the readback fails, so callers can
//! fall back to CPU blending. On wasm (browser WebGPU) the async
//! `new_async` / `blend_rgba_async` pair replaces it, since blocking the
//! event loop is not allowed there.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;

use wgpu::util::DeviceExt;

/// Workgroup tile is 8x8 threads; a plain 1D dispatch would exceed the
/// 65535-per-dimension workgroup limit at ~4.2 megapixels (4K and beyond).
const TILE: u32 = 8;

const SHADER: &str = r#"
struct Matrices {
    header: vec4<u32>, // .x = width, .y = height, rest padding (vec4 keeps the struct at 16-byte alignment)
    left: array<vec4<f32>, 3>,
    right: array<vec4<f32>, 3>,
};

@group(0) @binding(0) var<uniform> mats: Matrices;
@group(0) @binding(1) var<storage, read> left_frame: array<u32>;
@group(0) @binding(2) var<storage, read> right_frame: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_frame: array<u32>;

// Read one packed RGB pixel starting at byte offset `byte` from a dense
// RGB24 byte stream. Pixels may straddle two words, so read the second word
// as well when the pixel is not word aligned. The input buffers are padded
// with one extra zero word, so reading word + 1 is always in bounds.
// (Inlined at each use site: WGSL cannot pass storage pointers to functions.)
fn unpack_rgb32(packed: u32) -> vec3<f32> {
    return vec3<f32>(
        f32(packed & 0xFFu),
        f32((packed >> 8u) & 0xFFu),
        f32((packed >> 16u) & 0xFFu),
    );
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= mats.header.x || global_id.y >= mats.header.y) {
        return;
    }
    let i = global_id.y * mats.header.x + global_id.x;
    let byte = i * 3u;
    let word = byte >> 2u;
    let shift = (byte & 3u) * 8u;

    var l32: u32;
    if (shift == 0u) {
        l32 = left_frame[word];
    } else {
        l32 = (left_frame[word] >> shift) | (left_frame[word + 1u] << (32u - shift));
    }
    var r32: u32;
    if (shift == 0u) {
        r32 = right_frame[word];
    } else {
        r32 = (right_frame[word] >> shift) | (right_frame[word + 1u] << (32u - shift));
    }

    let l = unpack_rgb32(l32);
    let r = unpack_rgb32(r32);
    let out = vec3<f32>(
        dot(mats.left[0].xyz, l) + dot(mats.right[0].xyz, r),
        dot(mats.left[1].xyz, l) + dot(mats.right[1].xyz, r),
        dot(mats.left[2].xyz, l) + dot(mats.right[2].xyz, r),
    );
    // round + clamp matches the CPU reference (Rust f32::round is also
    // half-away-from-zero); u32() alone would truncate toward zero.
    let q = clamp(round(out), vec3<f32>(0.0), vec3<f32>(255.0));
    out_frame[i] = u32(q.x) | (u32(q.y) << 8u) | (u32(q.z) << 16u);
}

// RGBA variant (gifs): every pixel is exactly one 32-bit word, so no
// straddling unpacking is needed. Where both source pixels are fully
// transparent the output is transparent; otherwise the RGB channels are
// blended and alpha becomes opaque, matching the CPU reference.
@compute @workgroup_size(8, 8)
fn main_rgba(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= mats.header.x || global_id.y >= mats.header.y) {
        return;
    }
    let i = global_id.y * mats.header.x + global_id.x;
    let l32 = left_frame[i];
    let r32 = right_frame[i];
    if ((l32 & 0xFF000000u) == 0u) && ((r32 & 0xFF000000u) == 0u) {
        out_frame[i] = 0u;
        return;
    }
    let l = unpack_rgb32(l32);
    let r = unpack_rgb32(r32);
    let out = vec3<f32>(
        dot(mats.left[0].xyz, l) + dot(mats.right[0].xyz, r),
        dot(mats.left[1].xyz, l) + dot(mats.right[1].xyz, r),
        dot(mats.left[2].xyz, l) + dot(mats.right[2].xyz, r),
    );
    let q = clamp(round(out), vec3<f32>(0.0), vec3<f32>(255.0));
    out_frame[i] = u32(q.x) | (u32(q.y) << 8u) | (u32(q.z) << 16u) | 0xFF000000u;
}
"#;

/// Uniform block matching the WGSL `Matrices` struct (112 bytes).
#[derive(Clone, Copy)]
#[repr(C)]
struct Uniforms {
    width: u32,
    height: u32,
    _pad: [u32; 2],
    left: [[f32; 4]; 3],
    right: [[f32; 4]; 3],
}

// `to_bytes` hand-rolls this layout; fail fast if the struct ever drifts.
const _: () = assert!(core::mem::size_of::<Uniforms>() == 112);

/// Convert the anaglyph matrix (`[[f32; 9]; 2]`, row-major rows for the left
/// and right side) into the uniform buffer layout.
fn uniforms(width: u32, height: u32, matrix: &[[f32; 9]; 2]) -> Uniforms {
    let rows = |m: &[f32; 9]| -> [[f32; 4]; 3] {
        [
            [m[0], m[1], m[2], 0.0],
            [m[3], m[4], m[5], 0.0],
            [m[6], m[7], m[8], 0.0],
        ]
    };
    Uniforms {
        width,
        height,
        _pad: [0; 2],
        left: rows(&matrix[0]),
        right: rows(&matrix[1]),
    }
}

impl Uniforms {
    /// Uniform block byte layout. WGSL rounds the struct size up to its
    /// alignment (16), so the binding is 128 bytes even though the fields
    /// only occupy 112.
    fn to_bytes(self) -> [u8; 128] {
        let mut bytes = [0u8; 128];
        let mut pos = 0usize;
        let mut push = |v: u32| {
            bytes[pos..pos + 4].copy_from_slice(&v.to_le_bytes());
            pos += 4;
        };
        push(self.width);
        push(self.height);
        push(0);
        push(0);
        for side in [self.left, self.right] {
            for row in side {
                for v in row {
                    push(v.to_bits());
                }
            }
        }
        bytes
    }
}

/// GPU anaglyph blender.
///
/// Owns the wgpu device plus two ping-pong input buffers (so consecutive
/// blend calls do not overwrite buffers the GPU may still be reading) and one
/// output buffer with a matching CPU-readback buffer.
///
/// Two pipelines share the same buffers and bind groups: one consumes dense
/// RGB24 frames (video), the other dense RGBA frames (gifs). Input buffers
/// are sized for RGBA (`4 * pixels + 4`), which covers the RGB24 upload
/// padding as well.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub struct GpuBlender {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pipeline_rgba: wgpu::ComputePipeline,
    bind_groups: [wgpu::BindGroup; 2],
    frame_buffers: [wgpu::Buffer; 2],
    out_buffer: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Output byte count: 4 bytes per pixel.
    out_len: usize,
    /// Reusable staging buffer for padding RGB24 uploads to 4-byte multiples
    /// (`4 * pixels + 4`, including the sentinel word); its tail is zero and
    /// stays zero across uploads.
    staging: Vec<u8>,
    pixel_count: u32,
    /// 2D workgroup grid for the blend dispatch.
    dispatch: (u32, u32),
    parity: usize,
}

impl GpuBlender {
    /// Create a blender for frames of `width` x `height` using the given
    /// anaglyph matrix. Returns an error message when no usable GPU adapter
    /// is found or the frame is too large for device limits.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(width: u32, height: u32, matrix: &[[f32; 9]; 2]) -> Result<GpuBlender, String> {
        pollster::block_on(Self::build(width, height, matrix))
    }

    /// Async variant of [`Self::new`]. This is the only constructor on wasm,
    /// where blocking the thread is not allowed; on native it underlies the
    /// synchronous constructor.
    pub async fn new_async(
        width: u32,
        height: u32,
        matrix: &[[f32; 9]; 2],
    ) -> Result<GpuBlender, String> {
        Self::build(width, height, matrix).await
    }

    async fn build(width: u32, height: u32, matrix: &[[f32; 9]; 2]) -> Result<GpuBlender, String> {
        let pixel_count = width
            .checked_mul(height)
            .ok_or_else(|| format!("frame too large: {width}x{height}"))?;

        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no suitable GPU adapter found: {e}"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("anaglyph-gpu"),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("failed to create device: {e}"))?;

        let out_bytes = (pixel_count as u64) * 4;
        // Input buffers must hold either a padded RGB24 stream (3 bytes per
        // pixel, rounded up to words, plus a sentinel word for straddling
        // reads) or a full RGBA frame; the RGBA size covers both.
        let upload_len = (pixel_count as usize) * 4 + 4;
        let needed = (upload_len as u64).max(out_bytes);
        let max_storage = device.limits().max_storage_buffer_binding_size;
        if needed > max_storage {
            return Err(format!(
                "frame too large for GPU: needs {} bytes, device limit is {} bytes",
                needed, max_storage
            ));
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anaglyph"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let (dispatch_x, dispatch_y) = (width.div_ceil(TILE), height.div_ceil(TILE));
        let max_groups = device.limits().max_compute_workgroups_per_dimension;
        if dispatch_x > max_groups || dispatch_y > max_groups {
            return Err(format!(
                "frame too large for GPU: dispatch grid {dispatch_x}x{dispatch_y} \
                 exceeds per-dimension workgroup limit {max_groups}"
            ));
        }

        let uniforms = uniforms(width, height, matrix).to_bytes();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("anaglyph-matrices"),
            contents: &uniforms,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let storage_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let frame_buffers = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("anaglyph-in0"),
                size: upload_len as u64,
                usage: storage_usage,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("anaglyph-in1"),
                size: upload_len as u64,
                usage: storage_usage,
                mapped_at_creation: false,
            }),
        ];
        let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("anaglyph-out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("anaglyph-readback"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("anaglyph-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("anaglyph-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let make_pipeline = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipeline = make_pipeline("anaglyph-pipeline", "main");
        let pipeline_rgba = make_pipeline("anaglyph-pipeline-rgba", "main_rgba");

        // Two bind groups, one per ping-pong parity: binding 1 (left) always
        // points at the buffer written this frame, binding 2 (right) at the
        // other one. Both buffers are fully overwritten each call, so the
        // assignment of which side lives where does not matter.
        let bind_groups = [0, 1].map(|p| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("anaglyph-bg"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: frame_buffers[p].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: frame_buffers[p ^ 1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_buffer.as_entire_binding(),
                    },
                ],
            })
        });

        Ok(GpuBlender {
            device,
            queue,
            pipeline,
            pipeline_rgba,
            bind_groups,
            frame_buffers,
            out_buffer,
            readback,
            out_len: out_bytes as usize,
            staging: vec![0u8; upload_len],
            pixel_count,
            dispatch: (dispatch_x, dispatch_y),
            parity: 0,
        })
    }

    /// Blend `left` and `right` (dense RGB24, `3 * width * height` bytes each)
    /// on the GPU and return the result as a dense RGB24 vec.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn blend(&mut self, left: &[u8], right: &[u8]) -> Result<Vec<u8>, String> {
        let expected = (self.pixel_count as u64) * 3;
        if left.len() as u64 != expected || right.len() as u64 != expected {
            return Err(format!(
                "frame size mismatch: got {}/{} bytes, expected {expected}",
                left.len(),
                right.len()
            ));
        }

        let p = self.parity;
        self.parity ^= 1;

        self.upload(left, p);
        self.upload(right, p ^ 1);

        let words = self.run_and_read(p, false)?;

        // Unpad stride 4 -> stride 3.
        let mut out = vec![0u8; (self.pixel_count as usize) * 3];
        for (src, dst) in words
            .as_chunks::<4>()
            .0
            .iter()
            .zip(out.as_chunks_mut::<3>().0.iter_mut())
        {
            dst.copy_from_slice(&src[..3]);
        }
        Ok(out)
    }

    /// Blend `left` and `right` (dense RGBA, `4 * width * height` bytes each)
    /// on the GPU and return the result as a dense RGBA vec. Pixels where both
    /// inputs are fully transparent stay transparent; everything else becomes
    /// opaque.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn blend_rgba(&mut self, left: &[u8], right: &[u8]) -> Result<Vec<u8>, String> {
        let len = (self.pixel_count as usize) * 4;
        if left.len() != len || right.len() != len {
            return Err(format!(
                "frame size mismatch: got {}/{} bytes, expected {len}",
                left.len(),
                right.len()
            ));
        }

        let p = self.parity;
        self.parity ^= 1;

        // RGBA frames are already word-aligned; no staging padding needed.
        self.queue.write_buffer(&self.frame_buffers[p], 0, left);
        self.queue
            .write_buffer(&self.frame_buffers[p ^ 1], 0, right);

        self.run_and_read(p, true)
    }

    /// Async RGBA blend for wasm. The readback mapping resolves through the
    /// browser's event loop (WebGPU `mapAsync`), so nothing here blocks the
    /// main thread.
    #[cfg(target_arch = "wasm32")]
    pub async fn blend_rgba_async(&mut self, left: &[u8], right: &[u8]) -> Result<Vec<u8>, String> {
        let len = (self.pixel_count as usize) * 4;
        if left.len() != len || right.len() != len {
            return Err(format!(
                "frame size mismatch: got {}/{} bytes, expected {len}",
                left.len(),
                right.len()
            ));
        }

        let p = self.parity;
        self.parity ^= 1;

        self.queue.write_buffer(&self.frame_buffers[p], 0, left);
        self.queue
            .write_buffer(&self.frame_buffers[p ^ 1], 0, right);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anaglyph-encode"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("anaglyph-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_rgba);
            pass.set_bind_group(0, &self.bind_groups[p], &[]);
            pass.dispatch_workgroups(self.dispatch.0, self.dispatch.1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buffer, 0, &self.readback, 0, self.out_len as u64);
        self.queue.submit([encoder.finish()]);

        // The map_async callback is invoked when the WebGPU mapAsync promise
        // resolves on the JS event loop; awaiting the channel yields until
        // then without blocking.
        let (tx, rx) = futures_channel::oneshot::channel();
        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        rx.await
            .map_err(|_| "gpu readback wait cancelled".to_string())?
            .map_err(|e| format!("gpu readback mapping failed: {e}"))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| format!("gpu readback range: {e}"))?
            .to_vec();
        self.readback.unmap();
        Ok(mapped)
    }

    /// Dispatch the blend compute pass with the bind group for `parity`,
    /// copy the output into the readback buffer, and block until `out_len`
    /// bytes are readable as `4`-byte-per-pixel words.
    #[cfg(not(target_arch = "wasm32"))]
    fn run_and_read(&mut self, parity: usize, rgba: bool) -> Result<Vec<u8>, String> {
        let pipeline = if rgba {
            &self.pipeline_rgba
        } else {
            &self.pipeline
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anaglyph-encode"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("anaglyph-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind_groups[parity], &[]);
            pass.dispatch_workgroups(self.dispatch.0, self.dispatch.1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buffer, 0, &self.readback, 0, self.out_len as u64);
        self.queue.submit([encoder.finish()]);

        // Wait for the compute pass, then read the result back. The
        // map_async callback fires while `poll` waits, so `rx.recv` cannot
        // block indefinitely once poll returns; a disconnected channel or
        // an errored map means the device is gone — report, don't panic,
        // so callers can fall back to CPU blending.
        let (tx, rx) = mpsc::channel();
        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| format!("gpu poll failed: {e}"))?;
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(format!("gpu readback mapping failed: {e}")),
            Err(_) => return Err("gpu readback wait cancelled".into()),
        }
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| format!("gpu readback range: {e}"))?
            .to_vec();
        self.readback.unmap();
        Ok(mapped)
    }

    /// Stage one RGB24 frame into the padded upload buffer and copy it into
    /// `frame_buffers[target]`.
    #[cfg(not(target_arch = "wasm32"))]
    fn upload(&mut self, frame: &[u8], target: usize) {
        let len = frame.len();
        self.staging[..len].copy_from_slice(frame);
        // The tail past `len` is zero from construction and never written
        // (every RGB24 frame has the same length), so no re-padding needed.
        self.queue
            .write_buffer(&self.frame_buffers[target], 0, &self.staging);
    }
}

// The sync blend API exercised here is native-only; the wasm path is checked
// by compiling against the async entry points instead.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::anaglyph::combine_slices;

    /// Construct a blender or skip the test when no GPU adapter is
    /// available (headless CI), so `cargo test` stays green everywhere.
    fn blender_or_skip(width: u32, height: u32, matrix: &[[f32; 9]; 2]) -> Option<GpuBlender> {
        match GpuBlender::new(width, height, matrix) {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("skipping gpu test (no usable device): {e}");
                None
            }
        }
    }

    fn reference(left: &[u8], right: &[u8], matrix: &[[f32; 9]; 2]) -> Vec<u8> {
        let mut out = vec![0u8; left.len()];
        for i in 0..left.len() / 3 {
            combine_slices(
                &left[i * 3..i * 3 + 3],
                &right[i * 3..i * 3 + 3],
                &mut out[i * 3..i * 3 + 3],
                matrix,
            );
        }
        out
    }

    #[test]
    fn gpu_blend_matches_cpu_reference() {
        // Odd dimensions so pixels straddle 32-bit word boundaries.
        let (w, h) = (37usize, 11usize);
        let matrix = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let Some(mut gpu) = blender_or_skip(w as u32, h as u32, &matrix) else {
            return;
        };
        let n = w * h * 3;
        let left: Vec<u8> = (0..n).map(|i| (i.wrapping_mul(7) % 251) as u8).collect();
        let right: Vec<u8> = (0..n).map(|i| (i.wrapping_mul(13) % 199) as u8).collect();

        // Run twice to also exercise ping-pong buffer parity.
        for _ in 0..2 {
            let out = gpu.blend(&left, &right).expect("gpu blend");
            assert_eq!(out, reference(&left, &right, &matrix));
        }
    }

    /// CPU reference for the RGBA path: transparent where both inputs are,
    /// blended RGB with forced opaque alpha elsewhere.
    fn reference_rgba(left: &[u8], right: &[u8], matrix: &[[f32; 9]; 2]) -> Vec<u8> {
        let mut out = vec![0u8; left.len()];
        for i in 0..left.len() / 4 {
            let l = &left[i * 4..i * 4 + 4];
            let r = &right[i * 4..i * 4 + 4];
            if l[3] == 0 && r[3] == 0 {
                continue;
            }
            combine_slices(&l[..3], &r[..3], &mut out[i * 4..i * 4 + 3], matrix);
            out[i * 4 + 3] = 255;
        }
        out
    }

    #[test]
    fn gpu_blend_rgba_matches_cpu_reference() {
        let (w, h) = (13usize, 7usize);
        let matrix = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let Some(mut gpu) = blender_or_skip(w as u32, h as u32, &matrix) else {
            return;
        };
        let n = w * h * 4;
        // Sprinkle fully transparent pixels through both frames so the
        // both-transparent rule and the force-opaque rule are exercised.
        let left: Vec<u8> = (0..n)
            .map(|i| match i % 4 {
                3 if (i / 4) % 3 == 0 => 0,
                3 => 255,
                _ => (i.wrapping_mul(7) % 251) as u8,
            })
            .collect();
        let right: Vec<u8> = (0..n)
            .map(|i| match i % 4 {
                3 if (i / 4) % 2 == 0 => 0,
                3 => 255,
                _ => (i.wrapping_mul(13) % 199) as u8,
            })
            .collect();

        for _ in 0..2 {
            let out = gpu.blend_rgba(&left, &right).expect("gpu rgba blend");
            assert_eq!(out, reference_rgba(&left, &right, &matrix));
        }
    }

    /// The WGSL `Matrices` struct and the hand-rolled `to_bytes` writer must
    /// agree on offsets; a silent drift here garbles every blend.
    #[test]
    fn uniform_bytes_match_shader_layout() {
        let matrix = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
        ];
        let bytes = uniforms(0x11223344, 0x55667788, &matrix).to_bytes();

        assert_eq!(&bytes[0..4], &0x11223344u32.to_le_bytes()); // width
        assert_eq!(&bytes[4..8], &0x55667788u32.to_le_bytes()); // height
        assert_eq!(&bytes[8..16], &[0u8; 8]); // padding

        // Left rows start at byte 16; each row is 3 coefficients plus a 0
        // pad word (16-byte rows), so the last left value sits at 16+40=56.
        assert_eq!(&bytes[16..20], &1.0f32.to_le_bytes());
        assert_eq!(&bytes[20..24], &2.0f32.to_le_bytes());
        assert_eq!(&bytes[24..28], &3.0f32.to_le_bytes());
        assert_eq!(&bytes[28..32], &0.0f32.to_le_bytes());
        assert_eq!(&bytes[56..60], &9.0f32.to_le_bytes()); // last left value

        // Right side follows the left (16 + 3 * 16 = 64).
        assert_eq!(&bytes[64..68], &(-1.0f32).to_le_bytes());
        assert_eq!(&bytes[104..108], &(-9.0f32).to_le_bytes());
    }

    #[test]
    fn frame_too_large_is_an_error() {
        let matrix = crate::anaglyph::anaglyph_type_to_matrix(crate::anaglyph::AnaglyphType::Color);
        // u32::MAX x 2 overflows the pixel count.
        assert!(GpuBlender::new(u32::MAX, 2, matrix).is_err());
    }
}
