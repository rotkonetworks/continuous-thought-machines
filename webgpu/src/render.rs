//! 3D rendering of tick activation landscape.
//!
//! Renders a surface/point cloud where:
//!   X = tick index (0..K-1)
//!   Y = metric value (loss or selection %)
//!   Z = training step (time axis)
//!
//! Supports mouse rotation, zoom, and time-scrubbing.

use egui::{Color32, Painter, Pos2, Rect, Stroke};
use glam::{Mat4, Vec3, Vec4};

use crate::data::{Snapshot, SnapshotBuffer};

/// Camera state for 3D projection.
pub struct Camera {
    pub rotation_x: f32,
    pub rotation_y: f32,
    pub zoom: f32,
    pub fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            rotation_x: -0.4,
            rotation_y: 0.6,
            zoom: 1.0,
            fov: 400.0,
        }
    }
}

impl Camera {
    /// Project a 3D point to 2D screen coordinates.
    pub fn project(&self, point: Vec3, center: Pos2) -> Option<(Pos2, f32)> {
        let rot_y = Mat4::from_rotation_y(self.rotation_y);
        let rot_x = Mat4::from_rotation_x(self.rotation_x);
        let rotated = rot_x * rot_y * Vec4::new(point.x, point.y, point.z, 1.0);

        let depth = rotated.z + 5.0;
        if depth < 0.1 {
            return None;
        }

        let scale = self.fov * self.zoom / depth;
        let screen_x = center.x + rotated.x * scale;
        let screen_y = center.y + rotated.y * scale;

        Some((Pos2::new(screen_x, screen_y), depth))
    }
}

/// What metric to display.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorMode {
    Loss,
    Selection,
}

/// Visualization mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViewMode {
    Surface,
    Heatmap,
    Trajectories,
}

/// Map a value to a color.
pub fn value_to_color(val: f64, mode: ColorMode) -> Color32 {
    match mode {
        ColorMode::Loss => {
            // Low loss = green, high loss = red
            let t = ((val - 1.5) / 3.0).clamp(0.0, 1.0) as f32;
            Color32::from_rgb(
                (50.0 + t * 205.0) as u8,
                (220.0 - t * 180.0) as u8,
                50,
            )
        }
        ColorMode::Selection => {
            // 0% = dark, high% = bright cyan
            let t = (val / 25.0).clamp(0.0, 1.0) as f32;
            Color32::from_rgb(
                (20.0 + t * 40.0) as u8,
                (40.0 + t * 200.0) as u8,
                (80.0 + t * 175.0) as u8,
            )
        }
    }
}

/// Certainty to color (blue=uncertain, orange=certain).
pub fn certainty_color(cert: f64) -> Color32 {
    let t = cert.clamp(0.0, 1.0) as f32;
    Color32::from_rgb(
        (70.0 + t * 180.0) as u8,
        (170.0 - t * 80.0) as u8,
        (255.0 - t * 200.0) as u8,
    )
}

/// Which metric overlays to render as 3D lines.
pub struct Overlays {
    pub loss: bool,
    pub certainty: bool,
    pub tok_sec: bool,
    pub lr: bool,
    pub grad_frac: bool,
}

/// Render the 3D surface view.
pub fn render_surface(
    painter: &Painter,
    rect: Rect,
    buffer: &SnapshotBuffer,
    camera: &Camera,
    color_mode: ColorMode,
    window_size: usize,
    highlight_step: Option<u64>,
    overlays: &Overlays,
    window_end: Option<usize>,
) {
    if buffer.is_empty() {
        return;
    }

    let center = rect.center();
    let visible: Vec<&Snapshot> = buffer.window(window_size, window_end).collect();
    let n = visible.len();
    if n < 2 {
        return;
    }
    let k = buffer.k();

    // Draw axes
    let axes = [
        (Vec3::new(2.0, 0.0, 0.0), Color32::from_rgb(255, 80, 80), "tick"),
        (Vec3::new(0.0, -1.5, 0.0), Color32::from_rgb(80, 255, 80), "value"),
        (Vec3::new(0.0, 0.0, 2.0), Color32::from_rgb(80, 80, 255), "step"),
    ];
    if let Some((origin, _)) = camera.project(Vec3::ZERO, center) {
        for (dir, color, label) in &axes {
            if let Some((end, _)) = camera.project(*dir * 0.8, center) {
                painter.line_segment(
                    [origin, end],
                    Stroke::new(1.0, color.linear_multiply(0.3)),
                );
                painter.text(
                    end + egui::Vec2::new(5.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    label,
                    egui::FontId::monospace(10.0),
                    color.linear_multiply(0.6),
                );
            }
        }
    }

    // Draw surface
    for (si, snap) in visible.iter().enumerate() {
        let z = (si as f32 / n as f32) * 3.0 - 1.5;
        let is_highlighted = highlight_step.is_some_and(|hs| snap.step == hs);
        let is_last = si == n - 1;

        // Line connecting ticks
        let snap_k = snap.ticks.len();
        if snap_k < 2 {
            continue;
        }
        let mut points = Vec::with_capacity(snap_k);
        for ki in 0..snap_k {
            let tick = &snap.ticks[ki];
            let x = (ki as f32 / (snap_k - 1) as f32) * 3.0 - 1.5;
            let val = match color_mode {
                ColorMode::Loss => tick.loss,
                ColorMode::Selection => tick.selected_pct,
            };
            let y_norm = match color_mode {
                ColorMode::Loss => -(val / 5.0) as f32 * 1.5,
                ColorMode::Selection => -(val / 30.0) as f32 * 1.5,
            };

            if let Some((screen, _depth)) = camera.project(Vec3::new(x, y_norm, z), center) {
                points.push((screen, val, ki));
            }
        }

        // Draw connecting line
        if points.len() >= 2 {
            let line_alpha = if is_highlighted || is_last { 0.6 } else { 0.08 };
            let line_color = Color32::WHITE.linear_multiply(line_alpha);
            for w in points.windows(2) {
                painter.line_segment([w[0].0, w[1].0], Stroke::new(1.0, line_color));
            }
        }

        // Draw dots
        for (screen, val, _ki) in &points {
            let color = value_to_color(*val, color_mode);
            let alpha = if is_highlighted || is_last {
                1.0
            } else {
                0.3 + 0.5 * (si as f32 / n as f32)
            };
            let radius = if is_highlighted || is_last { 3.0 } else { 1.5 };
            painter.circle_filled(*screen, radius, color.linear_multiply(alpha));
        }
    }

    // Tick labels
    for ki in (0..k).step_by(4) {
        let x = (ki as f32 / (k - 1) as f32) * 3.0 - 1.5;
        if let Some((pos, _)) = camera.project(Vec3::new(x, 0.15, 1.6), center) {
            painter.text(
                pos,
                egui::Align2::CENTER_TOP,
                format!("t{ki}"),
                egui::FontId::monospace(9.0),
                Color32::from_gray(128),
            );
        }
    }

    // Step labels along the Z (time) axis
    {
        let first_step = visible.first().map(|s| s.step).unwrap_or(0);
        let last_step = visible.last().map(|s| s.step).unwrap_or(0);
        let step_range = last_step.saturating_sub(first_step).max(1);
        // Pick ~5 evenly spaced labels
        let num_labels = 5usize;
        for li in 0..=num_labels {
            let frac = li as f32 / num_labels as f32;
            let z = frac * 3.0 - 1.5;
            let step_val = first_step + (frac as f64 * step_range as f64) as u64;
            let x_pos = -1.6; // just left of the surface
            if let Some((pos, _)) = camera.project(Vec3::new(x_pos, 0.15, z), center) {
                painter.text(
                    pos,
                    egui::Align2::RIGHT_CENTER,
                    format!("{}", step_val),
                    egui::FontId::monospace(9.0),
                    Color32::from_gray(100),
                );
            }
        }
        // Arrow at the end of Z axis showing direction
        let arrow_z = 1.7;
        if let (Some((tip, _)), Some((base, _))) = (
            camera.project(Vec3::new(-1.6, 0.0, arrow_z), center),
            camera.project(Vec3::new(-1.6, 0.0, arrow_z - 0.3), center),
        ) {
            painter.line_segment(
                [base, tip],
                Stroke::new(1.5, Color32::from_rgb(80, 80, 255).linear_multiply(0.5)),
            );
            painter.text(
                tip + egui::Vec2::new(-5.0, 0.0),
                egui::Align2::RIGHT_CENTER,
                "step →",
                egui::FontId::monospace(9.0),
                Color32::from_rgb(80, 80, 255).linear_multiply(0.6),
            );
        }
    }

    // GPU metric overlays as 3D lines alongside the tick surface
    // Each metric gets its own X-lane to the right of the tick surface
    struct MetricLine {
        x_pos: f32,
        color: Color32,
        label: &'static str,
        max_val: f32,
        scientific: bool,
    }

    let mut metric_lines: Vec<(MetricLine, Vec<f32>)> = Vec::new();
    let mut lane = 0;

    if overlays.loss {
        let vals: Vec<f32> = visible.iter().map(|s| s.loss as f32).collect();
        metric_lines.push((MetricLine {
            x_pos: 1.7 + lane as f32 * 0.2,
            color: Color32::from_rgb(245, 91, 91),
            label: "loss",
            max_val: 5.0,
            scientific: false,
        }, vals));
        lane += 1;
    }
    if overlays.certainty {
        let vals: Vec<f32> = visible.iter().map(|s| s.certainty_mean as f32).collect();
        metric_lines.push((MetricLine {
            x_pos: 1.7 + lane as f32 * 0.2,
            color: Color32::from_rgb(91, 200, 245),
            label: "certainty",
            max_val: 1.0,
            scientific: false,
        }, vals));
        lane += 1;
    }
    if overlays.tok_sec {
        let vals: Vec<f32> = visible.iter().map(|s| s.tok_per_sec.unwrap_or(0.0) as f32).collect();
        let mv = vals.iter().cloned().fold(0.0_f32, f32::max).max(1.0);
        metric_lines.push((MetricLine {
            x_pos: 1.7 + lane as f32 * 0.2,
            color: Color32::from_rgb(61, 220, 132),
            label: "tok/s",
            max_val: mv,
            scientific: false,
        }, vals));
        lane += 1;
    }
    if overlays.lr {
        let vals: Vec<f32> = visible.iter().map(|s| s.lr.unwrap_or(0.0) as f32).collect();
        let mv = vals.iter().cloned().fold(0.0_f32, f32::max).max(1e-6);
        metric_lines.push((MetricLine {
            x_pos: 1.7 + lane as f32 * 0.2,
            color: Color32::from_rgb(245, 166, 35),
            label: "lr",
            max_val: mv,
            scientific: true,
        }, vals));
        lane += 1;
    }
    if overlays.grad_frac {
        let vals: Vec<f32> = visible.iter().map(|s| s.grad_tick_frac as f32).collect();
        metric_lines.push((MetricLine {
            x_pos: 1.7 + lane as f32 * 0.2,
            color: Color32::from_rgb(180, 130, 255),
            label: "grad%",
            max_val: 1.0,
            scientific: false,
        }, vals));
    }

    for (ml, vals) in &metric_lines {
        // Draw a faint vertical rail
        if let (Some((top, _)), Some((bot, _))) = (
            camera.project(Vec3::new(ml.x_pos, -1.5, -1.5), center),
            camera.project(Vec3::new(ml.x_pos, 0.0, -1.5), center),
        ) {
            painter.line_segment([top, bot], Stroke::new(0.5, ml.color.linear_multiply(0.15)));
        }

        // Draw the 3D metric line
        let mut prev_screen: Option<Pos2> = None;
        for (si, v) in vals.iter().enumerate() {
            let z = (si as f32 / n as f32) * 3.0 - 1.5;
            let y = -(v / ml.max_val).clamp(0.0, 1.0) * 1.5;
            if let Some((screen, _)) = camera.project(Vec3::new(ml.x_pos, y, z), center) {
                if let Some(prev) = prev_screen {
                    let alpha = 0.3 + 0.6 * (si as f32 / n as f32);
                    painter.line_segment([prev, screen], Stroke::new(1.5, ml.color.linear_multiply(alpha)));
                }
                prev_screen = Some(screen);
            }
        }

        // Label at the front
        if let Some((label_pos, _)) = camera.project(Vec3::new(ml.x_pos, 0.15, 1.6), center) {
            painter.text(
                label_pos,
                egui::Align2::CENTER_TOP,
                ml.label,
                egui::FontId::monospace(9.0),
                ml.color.linear_multiply(0.8),
            );
        }

        // Value label at the last point
        if let Some(last_val) = vals.last() {
            let z_last = (((n - 1) as f32) / n as f32) * 3.0 - 1.5;
            let y_last = -(last_val / ml.max_val).clamp(0.0, 1.0) * 1.5;
            if let Some((pos, _)) = camera.project(Vec3::new(ml.x_pos + 0.15, y_last, z_last), center) {
                painter.text(
                    pos,
                    egui::Align2::LEFT_CENTER,
                    if ml.scientific { format!("{:.1e}", last_val) } else { format!("{:.1}", last_val) },
                    egui::FontId::monospace(8.0),
                    ml.color,
                );
            }
        }
    }
}

/// Render the 2D heatmap view.
pub fn render_heatmap(
    painter: &Painter,
    rect: Rect,
    buffer: &SnapshotBuffer,
    color_mode: ColorMode,
    window_size: usize,
    highlight_step: Option<u64>,
) {
    if buffer.is_empty() {
        return;
    }

    let visible: Vec<&Snapshot> = buffer.tail(window_size).collect();
    let n = visible.len();
    if n == 0 {
        return;
    }
    let k = buffer.k();

    let margin = 60.0;
    let area = Rect::from_min_max(
        Pos2::new(rect.min.x + margin, rect.min.y + margin),
        Pos2::new(rect.max.x - margin, rect.max.y - margin),
    );

    let cell_w = area.width() / k as f32;
    let cell_h = area.height() / n as f32;

    for (si, snap) in visible.iter().enumerate() {
        let is_highlighted = highlight_step.is_some_and(|hs| snap.step == hs);

        let snap_k = snap.ticks.len();
        if snap_k == 0 {
            continue;
        }
        let snap_cell_w = area.width() / snap_k as f32;
        for ki in 0..snap_k {
            let tick = &snap.ticks[ki];
            let val = match color_mode {
                ColorMode::Loss => tick.loss,
                ColorMode::Selection => tick.selected_pct,
            };
            let color = value_to_color(val, color_mode);
            let alpha = if is_highlighted { 1.0 } else { 0.8 };

            let cell_rect = Rect::from_min_size(
                Pos2::new(area.min.x + ki as f32 * snap_cell_w, area.min.y + si as f32 * cell_h),
                egui::Vec2::new(snap_cell_w - 0.5, cell_h.max(1.0)),
            );
            painter.rect_filled(cell_rect, 0.0, color.linear_multiply(alpha));
        }
    }

    // Tick labels
    for ki in (0..k).step_by(4) {
        let x = area.min.x + ki as f32 * cell_w + cell_w / 2.0;
        painter.text(
            Pos2::new(x, area.min.y - 8.0),
            egui::Align2::CENTER_BOTTOM,
            format!("t{ki}"),
            egui::FontId::monospace(10.0),
            Color32::from_gray(128),
        );
    }

    // Step labels
    if let (Some(first), Some(last)) = (visible.first(), visible.last()) {
        painter.text(
            Pos2::new(area.min.x - 5.0, area.min.y),
            egui::Align2::RIGHT_TOP,
            format!("{}", first.step),
            egui::FontId::monospace(9.0),
            Color32::from_gray(100),
        );
        painter.text(
            Pos2::new(area.min.x - 5.0, area.max.y),
            egui::Align2::RIGHT_BOTTOM,
            format!("{}", last.step),
            egui::FontId::monospace(9.0),
            Color32::from_gray(100),
        );
    }
}

// ============================================================
// Hebbian adaptation visualization
// ============================================================

/// Render the Hebbian adaptation demo: accuracy curve + sync PCA scatter.
pub fn render_hebbian(
    painter: &Painter,
    rect: Rect,
    eval: &crate::hebbian::EvalState,
    camera: &Camera,
    show_base: bool,
    step: usize,
) {
    let center = rect.center();
    let n = eval.n_images;
    if n == 0 || step == 0 {
        return;
    }
    let show = step.min(n);

    // ─── Accuracy curve (3D: X=image index, Y=accuracy, Z=0) ────────
    let base_acc = eval.base_accuracy();

    // Draw axes
    let axes = [
        (Vec3::new(2.0, 0.0, 0.0), Color32::from_rgb(255, 80, 80), "image →"),
        (Vec3::new(0.0, -1.5, 0.0), Color32::from_rgb(80, 255, 80), "accuracy"),
        (Vec3::new(0.0, 0.0, 2.0), Color32::from_rgb(80, 80, 255), "sync PC1"),
    ];
    if let Some((origin, _)) = camera.project(Vec3::ZERO, center) {
        for (dir, color, label) in &axes {
            if let Some((end, _)) = camera.project(*dir * 0.8, center) {
                painter.line_segment([origin, end],
                    Stroke::new(1.0, color.linear_multiply(0.3)));
                painter.text(end + egui::Vec2::new(5.0, 0.0),
                    egui::Align2::LEFT_CENTER, label,
                    egui::FontId::monospace(10.0),
                    color.linear_multiply(0.6));
            }
        }
    }

    // Base accuracy reference line
    if show_base {
        let y_base = -base_acc * 1.5;
        if let (Some((p1, _)), Some((p2, _))) = (
            camera.project(Vec3::new(-1.5, y_base, -1.5), center),
            camera.project(Vec3::new(1.5, y_base, -1.5), center),
        ) {
            painter.line_segment([p1, p2],
                Stroke::new(1.0, Color32::from_rgb(150, 150, 150)));
            painter.text(p2 + egui::Vec2::new(5.0, 0.0),
                egui::Align2::LEFT_CENTER,
                format!("base {:.0}%", base_acc * 100.0),
                egui::FontId::monospace(9.0),
                Color32::from_gray(150));
        }
    }

    // Hebbian accuracy curve
    let mut prev_screen: Option<Pos2> = None;
    for i in 0..show {
        let x = (i as f32 / n as f32) * 3.0 - 1.5;
        let y = -eval.cumulative_acc[i] * 1.5;
        let z = -1.5; // flat on the back wall

        if let Some((screen, _)) = camera.project(Vec3::new(x, y, z), center) {
            let correct = eval.hebbian_correct[i];
            let color = if correct {
                Color32::from_rgb(46, 204, 113)
            } else {
                Color32::from_rgb(231, 76, 60)
            };

            // Line connecting points
            if let Some(prev) = prev_screen {
                let alpha = 0.3 + 0.7 * (i as f32 / show as f32);
                painter.line_segment([prev, screen],
                    Stroke::new(2.0, Color32::from_rgb(46, 204, 113)
                        .linear_multiply(alpha)));
            }

            // Point
            painter.circle_filled(screen, 3.0, color);
            prev_screen = Some(screen);
        }
    }

    // ─── Sync PCA scatter (3D: X=PCA1, Y=PCA2, Z=confidence) ────────
    // Only show images processed so far
    let pca_scale = 0.01; // scale PCA coords to fit in view
    for i in 0..show {
        let px = eval.sync_pca[i * 2] * pca_scale;
        let py = eval.sync_pca[i * 2 + 1] * pca_scale;
        // Z = position in sequence (spreads points along the depth axis)
        let pz = (i as f32 / n as f32) * 3.0 - 1.5;

        if let Some((screen, depth)) = camera.project(
            Vec3::new(px.clamp(-1.5, 1.5), py.clamp(-1.5, 0.0), pz), center,
        ) {
            let correct = eval.hebbian_correct[i];
            let base_ok = eval.base_correct[i];

            let color = match (base_ok, correct) {
                (false, true) => Color32::from_rgb(0, 255, 100),   // fixed by Hebbian — bright green
                (true, true) => Color32::from_rgb(80, 140, 220),   // both correct — blue
                (true, false) => Color32::from_rgb(255, 100, 0),   // broken by Hebbian — orange
                (false, false) => Color32::from_rgb(120, 50, 50),  // both wrong — dark red
            };

            let alpha = 0.3 + 0.6 * (i as f32 / show as f32);
            let size = if depth > 0.0 { 2.0 + 3.0 / depth } else { 2.0 };
            painter.circle_filled(screen, size, color.linear_multiply(alpha));
        }
    }

    // Legend
    let legend_x = rect.max.x - 140.0;
    let legend_y = rect.max.y - 80.0;
    let legends = [
        (Color32::from_rgb(0, 255, 100), "Fixed by Hebbian"),
        (Color32::from_rgb(80, 140, 220), "Both correct"),
        (Color32::from_rgb(255, 100, 0), "Broken"),
        (Color32::from_rgb(120, 50, 50), "Both wrong"),
    ];
    for (i, (color, label)) in legends.iter().enumerate() {
        let y = legend_y + i as f32 * 16.0;
        painter.circle_filled(Pos2::new(legend_x, y), 4.0, *color);
        painter.text(Pos2::new(legend_x + 10.0, y),
            egui::Align2::LEFT_CENTER, *label,
            egui::FontId::monospace(9.0), Color32::from_gray(180));
    }

    // Current accuracy label
    if show > 0 {
        let acc = eval.cumulative_acc[show - 1];
        let text = format!("{:.1}% ({}/{})", acc * 100.0, show, n);
        painter.text(
            Pos2::new(rect.center().x, rect.min.y + 20.0),
            egui::Align2::CENTER_TOP, text,
            egui::FontId::monospace(14.0),
            Color32::from_rgb(46, 204, 113));
    }
}

/// Render bound analysis as a 3D architectural diagram.
///
/// Shows the CTM as a 3D structure where you can SEE the bottleneck:
/// - Ticks along X-axis, each a vertical slice
/// - Synapse capacity as a wide translucent cylinder between ticks
/// - Synapse utilization as a thin bright core inside the cylinder
/// - Overthinking ticks glow red
/// - Best tick glows green
/// - Top-5 SVs shown as descending bars
pub fn render_bounds_3d(
    painter: &Painter,
    rect: Rect,
    bounds: &crate::data::BoundsInfo,
    camera: &Camera,
) {
    let center = rect.center();
    let n_ticks = bounds.n_ticks;
    if n_ticks == 0 { return; }

    // Axes
    if let Some((origin, _)) = camera.project(Vec3::ZERO, center) {
        let axes = [
            (Vec3::new(2.0, 0.0, 0.0), Color32::from_rgb(255, 80, 80), "tick →"),
            (Vec3::new(0.0, -2.0, 0.0), Color32::from_rgb(80, 255, 80), "capacity"),
            (Vec3::new(0.0, 0.0, 1.5), Color32::from_rgb(80, 80, 255), "SVs"),
        ];
        for (dir, color, label) in &axes {
            if let Some((end, _)) = camera.project(*dir * 0.8, center) {
                painter.line_segment([origin, end], Stroke::new(1.0, color.linear_multiply(0.3)));
                painter.text(end + egui::Vec2::new(5.0, 0.0),
                    egui::Align2::LEFT_CENTER, label,
                    egui::FontId::monospace(9.0), color.linear_multiply(0.5));
            }
        }
    }

    // ─── Synapse capacity vs utilization (the main visual) ──────
    // Capacity = wide translucent band, utilization = thin bright core
    let capacity_rank = bounds.synapse_rank_90 as f32;
    let util_rank = bounds.synapse_activation_rank as f32;
    let max_rank = capacity_rank.max(1.0);

    let capacity_height = 1.5;  // full height = full rank
    let util_height = capacity_height * (util_rank / max_rank);

    // Draw capacity band (wide, faint)
    for t in 0..n_ticks.min(50) {
        let x = (t as f32 / n_ticks as f32) * 3.0 - 1.5;
        let is_overthinking = t as usize >= bounds.best_tick &&
            (bounds.n_overthinking as f32 / n_ticks as f32) > 0.3;
        let is_best = t == bounds.best_tick;

        // Capacity bar (outer, translucent)
        if let (Some((top, _)), Some((bot, _))) = (
            camera.project(Vec3::new(x, -capacity_height, 0.0), center),
            camera.project(Vec3::new(x, 0.0, 0.0), center),
        ) {
            let cap_color = Color32::from_rgba_premultiplied(100, 100, 200, 30);
            painter.line_segment([top, bot], Stroke::new(12.0, cap_color));
        }

        // Utilization bar (inner, bright)
        if let (Some((top, _)), Some((bot, _))) = (
            camera.project(Vec3::new(x, -util_height, 0.0), center),
            camera.project(Vec3::new(x, 0.0, 0.0), center),
        ) {
            let util_color = if is_best {
                Color32::from_rgb(46, 204, 113)  // green — best tick
            } else if is_overthinking {
                Color32::from_rgb(231, 76, 60)   // red — overthinking
            } else {
                Color32::from_rgb(91, 138, 245)  // blue — normal
            };
            painter.line_segment([top, bot], Stroke::new(4.0, util_color));
        }

        // Tick label
        if t % 5 == 0 || is_best {
            if let Some((pos, _)) = camera.project(Vec3::new(x, 0.15, 0.0), center) {
                let label = if is_best { format!("t{}★", t) } else { format!("t{}", t) };
                painter.text(pos, egui::Align2::CENTER_TOP, label,
                    egui::FontId::monospace(8.0),
                    if is_best { Color32::from_rgb(46, 204, 113) } else { Color32::from_gray(100) });
            }
        }
    }

    // ─── SVD spectrum (back wall) ───────────────────────────────
    let svs = &bounds.synapse_top_svs;
    if !svs.is_empty() {
        let max_sv = svs[0] as f32;
        for (i, &sv) in svs.iter().enumerate().take(5) {
            let z = (i as f32 / 4.0) * 2.0 - 1.0;
            let height = (sv as f32 / max_sv) * 1.5;

            if let (Some((top, _)), Some((bot, _))) = (
                camera.project(Vec3::new(1.6, -height, z), center),
                camera.project(Vec3::new(1.6, 0.0, z), center),
            ) {
                let alpha = 1.0 - i as f32 * 0.15;
                painter.line_segment([top, bot],
                    Stroke::new(8.0, Color32::from_rgb(180, 100, 255).linear_multiply(alpha)));
                painter.text(bot + egui::Vec2::new(0.0, 5.0),
                    egui::Align2::CENTER_TOP,
                    format!("σ{}={:.0}", i+1, sv),
                    egui::FontId::monospace(8.0),
                    Color32::from_gray(120));
            }
        }
    }

    // ─── Key stats as 3D text ───────────────────────────────────
    if let Some((pos, _)) = camera.project(Vec3::new(-1.5, -1.8, -0.5), center) {
        painter.text(pos, egui::Align2::LEFT_TOP,
            format!("Utilization: {:.1}%  (rank {}/{})",
                bounds.synapse_utilization_pct,
                bounds.synapse_activation_rank,
                bounds.synapse_rank_90),
            egui::FontId::monospace(11.0),
            Color32::from_rgb(245, 166, 35));
    }

    if let Some((pos, _)) = camera.project(Vec3::new(-1.5, -2.0, -0.5), center) {
        painter.text(pos, egui::Align2::LEFT_TOP,
            format!("{}", bounds.bottleneck),
            egui::FontId::monospace(10.0),
            Color32::from_rgb(91, 138, 245));
    }
}

/// Render bound analysis overlay on the Hebbian view (2D fallback).
pub fn render_bounds_overlay(
    painter: &Painter,
    rect: Rect,
    bounds: &crate::data::BoundsInfo,
) {
    let x = rect.min.x + 10.0;
    let mut y = rect.max.y - 160.0;
    let bar_w = 120.0;
    let bar_h = 10.0;

    // Title
    painter.text(
        Pos2::new(x, y),
        egui::Align2::LEFT_TOP, "BOUND ANALYSIS",
        egui::FontId::monospace(10.0),
        Color32::from_rgb(245, 166, 35),
    );
    y += 16.0;

    // Synapse utilization gauge
    let util = bounds.synapse_utilization_pct as f32 / 100.0;
    painter.text(
        Pos2::new(x, y),
        egui::Align2::LEFT_TOP,
        format!("Synapse: {:.1}%", bounds.synapse_utilization_pct),
        egui::FontId::monospace(9.0),
        Color32::from_gray(180),
    );
    y += 14.0;

    // Bar background
    painter.rect_filled(
        Rect::from_min_size(Pos2::new(x, y), egui::Vec2::new(bar_w, bar_h)),
        2.0, Color32::from_gray(40),
    );
    // Bar fill — red if low utilization, green if high
    let util_color = if util < 0.1 {
        Color32::from_rgb(231, 76, 60)   // red — bad
    } else if util < 0.5 {
        Color32::from_rgb(245, 166, 35)  // orange — ok
    } else {
        Color32::from_rgb(46, 204, 113)  // green — good
    };
    painter.rect_filled(
        Rect::from_min_size(Pos2::new(x, y), egui::Vec2::new(bar_w * util.min(1.0), bar_h)),
        2.0, util_color,
    );
    painter.text(
        Pos2::new(x + bar_w + 5.0, y - 1.0),
        egui::Align2::LEFT_TOP,
        format!("rank {}/{}", bounds.synapse_activation_rank, bounds.synapse_rank_90),
        egui::FontId::monospace(8.0),
        Color32::from_gray(120),
    );
    y += 18.0;

    // Overthinking gauge
    let overthink_frac = bounds.n_overthinking as f32 / bounds.n_ticks.max(1) as f32;
    painter.text(
        Pos2::new(x, y),
        egui::Align2::LEFT_TOP,
        format!("Overthinking: {}/{} ticks", bounds.n_overthinking, bounds.n_ticks),
        egui::FontId::monospace(9.0),
        Color32::from_gray(180),
    );
    y += 14.0;
    painter.rect_filled(
        Rect::from_min_size(Pos2::new(x, y), egui::Vec2::new(bar_w, bar_h)),
        2.0, Color32::from_gray(40),
    );
    let ot_color = if overthink_frac > 0.5 {
        Color32::from_rgb(231, 76, 60)
    } else if overthink_frac > 0.2 {
        Color32::from_rgb(245, 166, 35)
    } else {
        Color32::from_rgb(46, 204, 113)
    };
    painter.rect_filled(
        Rect::from_min_size(Pos2::new(x, y), egui::Vec2::new(bar_w * overthink_frac, bar_h)),
        2.0, ot_color,
    );
    painter.text(
        Pos2::new(x + bar_w + 5.0, y - 1.0),
        egui::Align2::LEFT_TOP,
        format!("best tick: {}", bounds.best_tick),
        egui::FontId::monospace(8.0),
        Color32::from_gray(120),
    );
    y += 18.0;

    // Condition number
    let cond_color = if bounds.synapse_condition > 500.0 {
        Color32::from_rgb(231, 76, 60)
    } else {
        Color32::from_gray(150)
    };
    painter.text(
        Pos2::new(x, y),
        egui::Align2::LEFT_TOP,
        format!("Condition: {:.0}", bounds.synapse_condition),
        egui::FontId::monospace(9.0),
        cond_color,
    );
    y += 16.0;

    // Bottleneck
    painter.text(
        Pos2::new(x, y),
        egui::Align2::LEFT_TOP,
        &bounds.bottleneck,
        egui::FontId::monospace(9.0),
        Color32::from_rgb(91, 138, 245),
    );
}

// ============================================================
// QEC surface code visualization
// ============================================================

/// Render a surface code lattice with errors, syndromes, and decoder output.
pub fn render_qec(
    painter: &Painter,
    rect: Rect,
    code: &crate::qec::SurfaceCode,
    result: &crate::qec::SyndromeResult,
    prediction: Option<u8>,
    tick: usize,
    n_ticks: usize,
) {
    let d = code.d;
    let margin = 60.0;
    let area = Rect::from_min_max(
        Pos2::new(rect.min.x + margin, rect.min.y + margin + 30.0),
        Pos2::new(rect.min.x + margin + 400.0, rect.min.y + margin + 430.0),
    );

    let cell_w = area.width() / d as f32;
    let cell_h = area.height() / d as f32;

    // Title
    let label_names = ["I (no error)", "X logical", "Z logical", "Y logical"];
    let true_label = result.label as usize;
    painter.text(
        Pos2::new(area.center().x, rect.min.y + 15.0),
        egui::Align2::CENTER_TOP,
        format!("Surface Code d={d}  |  True: {}  |  Tick: {tick}/{n_ticks}",
                label_names[true_label.min(3)]),
        egui::FontId::monospace(13.0),
        Color32::WHITE,
    );

    // Draw data qubits
    for q in 0..code.n_data {
        let (cx, cy) = code.qubit_pos(q);
        let x = area.min.x + (cx + 0.5) * cell_w;
        let y = area.min.y + (cy + 0.5) * cell_h;

        let has_x_err = result.x_errors[q];
        let has_z_err = result.z_errors[q];

        let color = match (has_x_err, has_z_err) {
            (false, false) => Color32::from_rgb(60, 60, 80),    // no error — dark
            (true, false) => Color32::from_rgb(255, 80, 80),     // X error — red
            (false, true) => Color32::from_rgb(80, 80, 255),     // Z error — blue
            (true, true) => Color32::from_rgb(200, 80, 200),     // Y error — purple
        };

        let radius = cell_w.min(cell_h) * 0.3;
        painter.circle_filled(Pos2::new(x, y), radius, color);

        // Qubit label
        painter.text(
            Pos2::new(x, y),
            egui::Align2::CENTER_CENTER,
            format!("{q}"),
            egui::FontId::monospace(8.0),
            Color32::from_gray(200),
        );
    }

    // Draw stabilizer syndromes
    for si in 0..code.n_stab {
        let (cx, cy) = code.stab_pos(si);
        let x = area.min.x + (cx + 0.5) * cell_w;
        let y = area.min.y + (cy + 0.5) * cell_h;

        let fired = result.syndrome[si];
        let is_x_stab = si < code.n_x_stab;

        if fired {
            let color = if is_x_stab {
                Color32::from_rgb(255, 200, 50)  // X-stabilizer fired — yellow
            } else {
                Color32::from_rgb(50, 255, 200)  // Z-stabilizer fired — cyan
            };
            let size = cell_w.min(cell_h) * 0.15;
            painter.rect_filled(
                Rect::from_center_size(Pos2::new(x, y), egui::Vec2::splat(size * 2.0)),
                2.0, color,
            );
        }
    }

    // Prediction display
    let pred_x = area.max.x + 40.0;
    let pred_y = area.min.y;

    if let Some(pred) = prediction {
        let pred_label = pred as usize;
        let correct = pred_label == true_label;
        let color = if correct {
            Color32::from_rgb(46, 204, 113)
        } else {
            Color32::from_rgb(231, 76, 60)
        };

        painter.text(
            Pos2::new(pred_x, pred_y),
            egui::Align2::LEFT_TOP,
            format!("Prediction: {}", label_names[pred_label.min(3)]),
            egui::FontId::monospace(14.0),
            color,
        );

        painter.text(
            Pos2::new(pred_x, pred_y + 20.0),
            egui::Align2::LEFT_TOP,
            if correct { "✓ CORRECT" } else { "✗ WRONG" },
            egui::FontId::monospace(16.0),
            color,
        );
    }

    // Thinking progress bar
    if n_ticks > 0 {
        let bar_y = area.max.y + 20.0;
        let bar_w = area.width();
        let bar_h = 12.0;
        let progress = tick as f32 / n_ticks as f32;

        // Background
        painter.rect_filled(
            Rect::from_min_size(
                Pos2::new(area.min.x, bar_y),
                egui::Vec2::new(bar_w, bar_h)),
            3.0, Color32::from_gray(40),
        );
        // Fill
        painter.rect_filled(
            Rect::from_min_size(
                Pos2::new(area.min.x, bar_y),
                egui::Vec2::new(bar_w * progress, bar_h)),
            3.0, Color32::from_rgb(91, 138, 245),
        );
        painter.text(
            Pos2::new(area.min.x + bar_w + 10.0, bar_y),
            egui::Align2::LEFT_TOP,
            format!("Thinking... {tick}/{n_ticks}"),
            egui::FontId::monospace(10.0),
            Color32::from_gray(150),
        );
    }

    // Legend
    let leg_x = pred_x;
    let leg_y = pred_y + 60.0;
    let legends = [
        (Color32::from_rgb(60, 60, 80), "No error"),
        (Color32::from_rgb(255, 80, 80), "X error"),
        (Color32::from_rgb(80, 80, 255), "Z error"),
        (Color32::from_rgb(200, 80, 200), "Y error"),
        (Color32::from_rgb(255, 200, 50), "X-stab fired"),
        (Color32::from_rgb(50, 255, 200), "Z-stab fired"),
    ];
    for (i, (color, label)) in legends.iter().enumerate() {
        let y = leg_y + i as f32 * 18.0;
        painter.circle_filled(Pos2::new(leg_x, y + 6.0), 5.0, *color);
        painter.text(
            Pos2::new(leg_x + 12.0, y),
            egui::Align2::LEFT_TOP, *label,
            egui::FontId::monospace(10.0),
            Color32::from_gray(180),
        );
    }
}
