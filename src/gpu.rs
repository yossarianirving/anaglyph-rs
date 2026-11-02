//! GPU-accelerated anaglyph frame blending using a wgpu compute shader.
//!
//! Combining two RGB24 frames into an anaglyph is a per-pixel linear
//! transform: `out[c] = dot(M_left[c], left.rgb) + dot(M_right[c], right.rgb)`.
//! This is embarrassingly parallel, so we run it on the GPU instead of a
//! scalar loop on the CPU.
//!
//! Frames are exchanged as dense RGB24 byte buffers (`3 * width * height`).
//! On the GPU, input pixels are unpacked from the byte stream directly
//! (each pixel may span two 32-bit words) and output pixels are written as
//! packed `0x00BBGGRR` words, which map to a byte order of `R, G, B, 0` in
//! memory, so the readback only needs a stride 4 -> 3 unpad.

use std::sync::mpsc;

use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 64;

const SHADER: &str = r#"
struct Matrices {
    header: vec4<u32>, // .x = num_pixels, rest padding (vec4 keeps the struct at 16-byte alignment)
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= mats.header.x) {
        return;
    }
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
    let clamped = clamp(out, vec3<f32>(0.0), vec3<f32>(255.0));
    out_frame[i] =
        u32(clamped.x) | (u32(clamped.y) << 8u) | (u32(clamped.z) << 16u);
}
"#;

/// Uniform block matching the WGSL `Matrices` struct (112 bytes).
#[derive(Clone, Copy)]
#[repr(C)]
struct Uniforms {
    num_pixels: u32,
    _pad: [u32; 3],
    left: [[f32; 4]; 3],
    right: [[f32; 4]; 3],
}

/// Convert the anaglyph matrix (`[[f32; 9]; 2]`, row-major rows for the left
/// and right side) into the uniform buffer layout.
fn uniforms(num_pixels: u32, matrix: &[[f32; 9]; 2]) -> Uniforms {
    let rows = |m: &[f32; 9]| -> [[f32; 4]; 3] {
        [
            [m[0], m[1], m[2], 0.0],
            [m[3], m[4], m[5], 0.0],
            [m[6], m[7], m[8], 0.0],
        ]
    };
    Uniforms {
        num_pixels,
        _pad: [0; 3],
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
        push(self.num_pixels);
        push(0);
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
pub struct GpuBlender {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_groups: [wgpu::BindGroup; 2],
    frame_buffers: [wgpu::Buffer; 2],
    out_buffer: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Total upload length: RGB24 byte count rounded up to 32-bit words plus
    /// one sentinel word (so straddling reads are in bounds).
    upload_len: usize,
    /// Output byte count: 4 bytes per pixel.
    out_len: usize,
    /// Reusable staging buffer for padding uploads to 4-byte multiples.
    staging: Vec<u8>,
    pixel_count: u32,
    workgroups: u32,
    parity: usize,
}

impl GpuBlender {
    /// Create a blender for frames of `width` x `height` using the given
    /// anaglyph matrix. Returns an error message when no usable GPU adapter
    /// is found or the frame is too large for device limits.
    pub fn new(
        width: u32,
        height: u32,
        matrix: &[[f32; 9]; 2],
    ) -> Result<GpuBlender, String> {
        let pixel_count = width
            .checked_mul(height)
            .ok_or_else(|| format!("frame too large: {width}x{height}"))?;

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| format!("no suitable GPU adapter found: {e}"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("anaglyph-gpu"),
                ..Default::default()
            },
        ))
        .map_err(|e| format!("failed to create device: {e}"))?;

        let frame_bytes = (pixel_count as u64) * 3;
        let out_bytes = (pixel_count as u64) * 4;
        // Upload length: RGB24 byte count padded up to 32-bit words plus one
        // sentinel word (so straddling reads are in bounds).
        let upload_len = (frame_bytes as usize).div_ceil(4) * 4 + 4;
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

        let uniforms = uniforms(pixel_count, matrix).to_bytes();
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

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("anaglyph-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

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
            bind_groups,
            frame_buffers,
            out_buffer,
            readback,
            upload_len,
            out_len: out_bytes as usize,
            staging: vec![0u8; upload_len],
            pixel_count,
            workgroups: pixel_count.div_ceil(WORKGROUP_SIZE),
            parity: 0,
        })
    }

    /// Blend `left` and `right` (dense RGB24, `3 * width * height` bytes each)
    /// on the GPU and return the result as a dense RGB24 vec.
    pub fn blend(&mut self, left: &[u8], right: &[u8]) -> Vec<u8> {
        debug_assert_eq!(left.len() as u64, (self.pixel_count as u64) * 3);
        debug_assert_eq!(right.len() as u64, (self.pixel_count as u64) * 3);

        let p = self.parity;
        self.parity ^= 1;

        self.upload(left, p);
        self.upload(right, p ^ 1);

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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_groups[p], &[]);
            pass.dispatch_workgroups(self.workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buffer, 0, &self.readback, 0, self.out_len as u64);
        self.queue.submit([encoder.finish()]);

        // Wait for the compute pass, then read the result back.
        let (tx, rx) = mpsc::channel();
        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => panic!("gpu readback mapping failed: {e}"),
            Err(e) => panic!("gpu readback wait failed: {e}"),
        }
        let mapped = slice.get_mapped_range().expect("gpu readback range").to_vec();
        self.readback.unmap();

        // Unpad stride 4 -> stride 3.
        let mut out = vec![0u8; (self.pixel_count as usize) * 3];
        for (src, dst) in mapped
            .chunks_exact(4)
            .zip(out.chunks_exact_mut(3))
        {
            dst.copy_from_slice(&src[..3]);
        }
        out
    }

    /// Stage one RGB24 frame into the padded upload buffer and copy it into
    /// `frame_buffers[target]`.
    fn upload(&mut self, frame: &[u8], target: usize) {
        let len = frame.len();
        self.staging[..len].copy_from_slice(frame);
        self.staging[len..self.upload_len].fill(0);
        self.queue
            .write_buffer(&self.frame_buffers[target], 0, &self.staging);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anaglyph::combine_slices;

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
        let n = w * h * 3;
        let left: Vec<u8> = (0..n).map(|i| (i.wrapping_mul(7) % 251) as u8).collect();
        let right: Vec<u8> = (0..n).map(|i| (i.wrapping_mul(13) % 199) as u8).collect();

        let mut gpu = GpuBlender::new(w as u32, h as u32, &matrix).expect("gpu available");
        // Run twice to also exercise ping-pong buffer parity.
        for _ in 0..2 {
            let out = gpu.blend(&left, &right);
            assert_eq!(out, reference(&left, &right, &matrix));
        }
    }
}
