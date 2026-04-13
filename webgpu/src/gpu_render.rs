//! Real wgpu 3D renderer for CTM bound analysis visualization.
//!
//! Renders neurons as instanced spheres, synapses as cylinders,
//! with proper depth buffer, lighting, and transparency.
//!
//! Integrates with egui via eframe's wgpu callback mechanism.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Vertex for instanced rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Per-instance data for neurons/bars.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Instance {
    pub model_col0: [f32; 4],
    pub model_col1: [f32; 4],
    pub model_col2: [f32; 4],
    pub model_col3: [f32; 4],
    pub color: [f32; 4],
}

/// Uniform buffer for camera.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub light_dir: [f32; 4],
}

/// A bar/cylinder instance for the bounds visualization.
pub struct BoundsBar {
    pub position: Vec3,
    pub height: f32,
    pub width: f32,
    pub color: [f32; 4],
}

/// Generate a unit box mesh (for bars/cylinders).
pub fn unit_box_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let p = 0.5_f32;
    let vertices = vec![
        // Front face
        Vertex { position: [-p, -p,  p], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [ p, -p,  p], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [ p,  p,  p], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [-p,  p,  p], normal: [0.0, 0.0, 1.0] },
        // Back face
        Vertex { position: [-p, -p, -p], normal: [0.0, 0.0, -1.0] },
        Vertex { position: [-p,  p, -p], normal: [0.0, 0.0, -1.0] },
        Vertex { position: [ p,  p, -p], normal: [0.0, 0.0, -1.0] },
        Vertex { position: [ p, -p, -p], normal: [0.0, 0.0, -1.0] },
        // Top face
        Vertex { position: [-p,  p, -p], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [-p,  p,  p], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [ p,  p,  p], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [ p,  p, -p], normal: [0.0, 1.0, 0.0] },
        // Bottom face
        Vertex { position: [-p, -p, -p], normal: [0.0, -1.0, 0.0] },
        Vertex { position: [ p, -p, -p], normal: [0.0, -1.0, 0.0] },
        Vertex { position: [ p, -p,  p], normal: [0.0, -1.0, 0.0] },
        Vertex { position: [-p, -p,  p], normal: [0.0, -1.0, 0.0] },
        // Right face
        Vertex { position: [ p, -p, -p], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [ p,  p, -p], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [ p,  p,  p], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [ p, -p,  p], normal: [1.0, 0.0, 0.0] },
        // Left face
        Vertex { position: [-p, -p, -p], normal: [-1.0, 0.0, 0.0] },
        Vertex { position: [-p, -p,  p], normal: [-1.0, 0.0, 0.0] },
        Vertex { position: [-p,  p,  p], normal: [-1.0, 0.0, 0.0] },
        Vertex { position: [-p,  p, -p], normal: [-1.0, 0.0, 0.0] },
    ];

    let indices: Vec<u16> = vec![
        0,1,2, 0,2,3,       // front
        4,5,6, 4,6,7,       // back
        8,9,10, 8,10,11,    // top
        12,13,14, 12,14,15, // bottom
        16,17,18, 16,18,19, // right
        20,21,22, 20,22,23, // left
    ];

    (vertices, indices)
}

/// WGSL shader for instanced 3D rendering with basic lighting.
pub const SHADER_SOURCE: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct InstanceInput {
    @location(2) model_col0: vec4<f32>,
    @location(3) model_col1: vec4<f32>,
    @location(4) model_col2: vec4<f32>,
    @location(5) model_col3: vec4<f32>,
    @location(6) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(
        instance.model_col0,
        instance.model_col1,
        instance.model_col2,
        instance.model_col3,
    );
    let world_pos = model * vec4<f32>(vertex.position, 1.0);
    let world_normal = normalize((model * vec4<f32>(vertex.normal, 0.0)).xyz);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_pos;
    out.world_normal = world_normal;
    out.color = instance.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(camera.light_dir.xyz);
    let ndotl = max(dot(in.world_normal, light_dir), 0.0);
    let ambient = 0.3;
    let diffuse = ndotl * 0.7;
    let lit = in.color.rgb * (ambient + diffuse);
    return vec4<f32>(lit, in.color.a);
}
"#;

/// Build instance data for bounds visualization bars.
pub fn build_bounds_instances(bounds: &crate::data::BoundsInfo) -> Vec<Instance> {
    let mut instances = Vec::new();
    let n_ticks = bounds.n_ticks;
    let capacity = bounds.synapse_rank_90 as f32;
    let utilization = bounds.synapse_activation_rank as f32;
    let max_rank = capacity.max(1.0);

    for t in 0..n_ticks.min(50) {
        let x = (t as f32 / n_ticks as f32) * 4.0 - 2.0;
        let is_best = t == bounds.best_tick;
        let is_overthinking = t > bounds.best_tick;

        // Capacity bar (outer, translucent)
        let cap_height = 2.0;
        let cap_width = 0.15;
        let cap_model = Mat4::from_translation(Vec3::new(x, cap_height * 0.5, 0.0))
            * Mat4::from_scale(Vec3::new(cap_width, cap_height, cap_width));
        let cols = cap_model.to_cols_array_2d();
        instances.push(Instance {
            model_col0: cols[0], model_col1: cols[1],
            model_col2: cols[2], model_col3: cols[3],
            color: [0.3, 0.3, 0.6, 0.15],
        });

        // Utilization bar (inner, bright)
        let util_height = cap_height * (utilization / max_rank);
        let util_width = cap_width * 0.4;
        let util_model = Mat4::from_translation(Vec3::new(x, util_height * 0.5, 0.0))
            * Mat4::from_scale(Vec3::new(util_width, util_height.max(0.01), util_width));
        let cols = util_model.to_cols_array_2d();
        let color = if is_best {
            [0.18, 0.8, 0.44, 1.0]  // green
        } else if is_overthinking && bounds.n_overthinking > n_ticks / 3 {
            [0.9, 0.3, 0.24, 1.0]   // red
        } else {
            [0.36, 0.54, 0.96, 1.0]  // blue
        };
        instances.push(Instance {
            model_col0: cols[0], model_col1: cols[1],
            model_col2: cols[2], model_col3: cols[3],
            color,
        });
    }

    // SVD bars on the right
    for (i, &sv) in bounds.synapse_top_svs.iter().enumerate().take(5) {
        let max_sv = bounds.synapse_top_svs[0];
        let z = (i as f32 / 4.0) * 3.0 - 1.5;
        let height = (sv / max_sv) as f32 * 2.0;
        let width = 0.2;

        let model = Mat4::from_translation(Vec3::new(2.5, height * 0.5, z))
            * Mat4::from_scale(Vec3::new(width, height, width));
        let cols = model.to_cols_array_2d();
        let alpha = 1.0 - i as f32 * 0.15;
        instances.push(Instance {
            model_col0: cols[0], model_col1: cols[1],
            model_col2: cols[2], model_col3: cols[3],
            color: [0.7 * alpha, 0.4 * alpha, 1.0 * alpha, 1.0],
        });
    }

    // Ground plane marker
    let ground = Mat4::from_translation(Vec3::new(0.0, -0.02, 0.0))
        * Mat4::from_scale(Vec3::new(5.0, 0.02, 4.0));
    let cols = ground.to_cols_array_2d();
    instances.push(Instance {
        model_col0: cols[0], model_col1: cols[1],
        model_col2: cols[2], model_col3: cols[3],
        color: [0.1, 0.1, 0.15, 0.5],
    });

    instances
}
