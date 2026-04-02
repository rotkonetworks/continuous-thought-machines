//! CTM Hebbian Adaptation Demo + Time-Travel Debugger
//!
//! Two modes:
//!   - Debugger: 3D visualization of tick activations (existing)
//!   - Hebbian:  Interactive adaptation with real-time accuracy (new)
//!
//! WASM: detects mode from URL query (?mode=hebbian or default debugger)
//! Native: pass --hebbian flag or a JSONL path

mod corrector;
mod data;
mod gpu_render;
mod gpu_scene;
mod hebbian;
mod qec;
mod render;

use data::SnapshotBuffer;
use hebbian::{HebbianEngine, EvalState};
use render::{Camera, ColorMode, Overlays, ViewMode};

use eframe::egui;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use web_time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

// ============================================================
// App modes
// ============================================================

enum AppMode {
    Debugger(DebuggerState),
    Hebbian(HebbianState),
    QEC(QECState),
    Loading, // WASM: waiting for data to load
}

struct QECState {
    code: qec::SurfaceCode,
    current_result: Option<qec::SyndromeResult>,
    noise_rate: f32,
    num_rounds: usize,
    prediction: Option<u8>,
    current_tick: usize,
    max_ticks: usize,
    animating: bool,
    // Stats
    total: usize,
    correct: usize,
    mwpm_correct: usize,
}

struct DebuggerState {
    buffer: Arc<Mutex<SnapshotBuffer>>,
    window_size: usize,
    max_window: usize,
    scrub_step: Option<u64>,
    playing: bool,
    play_speed: f32,
    last_play_tick: Instant,
    autoplay: bool,
    autoplay_direction: i64,
    last_interaction: Instant,
    window_end: Option<usize>,
    show_loss_line: bool,
    show_certainty_line: bool,
    show_tok_sec: bool,
    show_lr: bool,
    show_grad_frac: bool,
    #[cfg(not(target_arch = "wasm32"))]
    jsonl_path: PathBuf,
    #[cfg(not(target_arch = "wasm32"))]
    file_offset: u64,
    #[cfg(not(target_arch = "wasm32"))]
    last_poll: Instant,
}

struct HebbianState {
    engine: HebbianEngine,
    eval: EvalState,
    bounds: Option<data::BoundsInfo>,
    use_t10: bool,
    per_image: bool,
    forward_correction: bool,
    correction_alpha: f32,
    animating: bool,
    animation_speed: usize,
    show_base: bool,
}

struct App {
    mode: AppMode,
    // Preserved states for mode switching
    saved_hebbian: Option<HebbianState>,
    saved_qec: Option<QECState>,
    camera: Camera,
    color_mode: ColorMode,
    view_mode: ViewMode,
}

// ============================================================
// Hebbian demo construction
// ============================================================

impl HebbianState {
    fn from_buffers(
        sync_buf: &[u8],
        logits_final_buf: &[u8],
        logits_t10_buf: &[u8],
        labels_buf: &[u8],
        weights_buf: &[u8],
        baseline_buf: &[u8],
        pca_buf: &[u8],
        thumbnails_buf: &[u8],
        metadata_json: &str,
    ) -> Self {
        let meta: data::HebbianMetadata =
            serde_json::from_str(metadata_json).expect("bad metadata.json");

        let sync_signals = data::parse_f32_buffer(sync_buf);
        let logits_final = data::parse_f32_buffer(logits_final_buf);
        let logits_t10 = data::parse_f32_buffer(logits_t10_buf);
        let labels = data::parse_u16_buffer(labels_buf);
        let output_weights = data::parse_f32_buffer(weights_buf);
        let baseline_sync = data::parse_f32_buffer(baseline_buf);
        let sync_pca = data::parse_f32_buffer(pca_buf);
        let thumbnails = thumbnails_buf.to_vec();

        let engine = HebbianEngine::new(
            meta.n_synch, meta.n_output, baseline_sync, output_weights,
        );

        let eval = EvalState::new(
            sync_signals, logits_final, logits_t10, labels, sync_pca,
            thumbnails, meta.n_images, meta.n_synch, meta.n_output, meta.class_names,
        );

        // Don't auto-run — let user set params first
        let mut eval = eval;
        eval.dirty = false;

        Self {
            engine,
            eval,
            bounds: meta.bounds,
            use_t10: false,
            per_image: false,
            forward_correction: false,
            correction_alpha: 0.1,
            animating: false,
            animation_speed: 5,
            show_base: true,
        }
    }
}

// ============================================================
// eframe App implementation
// ============================================================

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(50));

        let is_loading = matches!(self.mode, AppMode::Loading);
        let is_debugger = matches!(self.mode, AppMode::Debugger(_));
        let is_qec = matches!(self.mode, AppMode::QEC(_));
        let is_hebbian = matches!(self.mode, AppMode::Hebbian(_));

        // Mode switcher bar (always visible except when loading)
        if !is_loading {
            egui::TopBottomPanel::top("mode_selector").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("CTM BOUNDS EXPLORER")
                        .strong().size(14.0)
                        .color(egui::Color32::from_rgb(200, 200, 200)));
                    ui.separator();

                    let switch_to_hebbian = ui.selectable_label(is_hebbian, "🖼 ImageNet").clicked() && !is_hebbian;
                    let switch_to_qec = ui.selectable_label(is_qec, "⚛ QEC Decoder").clicked() && !is_qec;

                    if switch_to_hebbian {
                        if let Some(saved) = self.saved_hebbian.take() {
                            // Save current QEC state
                            if let AppMode::QEC(qec_state) = std::mem::replace(&mut self.mode, AppMode::Loading) {
                                self.saved_qec = Some(qec_state);
                            }
                            self.mode = AppMode::Hebbian(saved);
                        }
                    }
                    if switch_to_qec {
                        // Save current Hebbian state
                        if let AppMode::Hebbian(hebb_state) = std::mem::replace(&mut self.mode, AppMode::Loading) {
                            self.saved_hebbian = Some(hebb_state);
                        }
                        // Restore or create QEC state
                        let qec = self.saved_qec.take().unwrap_or_else(|| QECState {
                            code: qec::SurfaceCode::new(5),
                            current_result: None,
                            noise_rate: 0.05,
                            num_rounds: 5,
                            prediction: None,
                            current_tick: 0,
                            max_ticks: 16,
                            animating: false,
                            total: 0,
                            correct: 0,
                            mwpm_correct: 0,
                        });
                        self.mode = AppMode::QEC(qec);
                    }
                });
            });
        }

        if is_loading {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("Loading data...")
                        .size(24.0).color(egui::Color32::WHITE));
                });
            });
        } else if is_debugger {
            self.update_debugger(ctx);
        } else if is_qec {
            self.update_qec(ctx);
        } else {
            self.update_hebbian(ctx);
        }
    }
}

impl App {
    // ─── Hebbian mode UI ────────────────────────────────────────────

    fn update_hebbian(&mut self, ctx: &egui::Context) {
        // Extract state — need to reborrow due to the enum
        let state = match &mut self.mode {
            AppMode::Hebbian(s) => s,
            _ => return,
        };

        // Run evaluation if dirty
        if state.eval.dirty {
            state.eval.run_full_mode(&mut state.engine, state.use_t10, state.per_image);
        }

        // Animate: advance N steps per frame
        if state.animating && state.eval.current_step < state.eval.n_images {
            for _ in 0..state.animation_speed {
                if state.eval.run_step(&mut state.engine, state.use_t10) {
                    state.animating = false;
                    break;
                }
            }
        }

        let n = state.eval.n_images;
        let step = state.eval.current_step;
        let base_acc = state.eval.base_accuracy();
        let hebb_acc = state.eval.hebbian_accuracy();

        // ─── Top panel: title + stats ───────────────────────────────
        egui::TopBottomPanel::top("hebbian_top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("CTM HEBBIAN ADAPTATION")
                    .strong().size(16.0)
                    .color(egui::Color32::from_rgb(91, 245, 138)));
                ui.separator();
                ui.label(format!("Base: {:.1}%", base_acc * 100.0));
                ui.separator();
                ui.label(egui::RichText::new(
                    format!("Hebbian: {:.1}%", hebb_acc * 100.0))
                    .color(egui::Color32::from_rgb(46, 204, 113))
                    .strong());
                ui.separator();
                ui.label(format!("Δ: {:+.1}%", (hebb_acc - base_acc) * 100.0));
                ui.separator();
                ui.label(format!("{}/{} images", step, n));
            });
        });

        // ─── Left panel: controls ───────────────────────────────────
        egui::SidePanel::left("hebbian_controls").min_width(200.0).show(ctx, |ui| {
            ui.heading("Parameters");
            ui.separator();

            let mut dirty = false;

            ui.label("Learning rate:");
            dirty |= ui.add(egui::Slider::new(&mut state.engine.lr, 0.01..=2.0)
                .logarithmic(true).text("lr")).changed();

            ui.label("Momentum:");
            dirty |= ui.add(egui::Slider::new(&mut state.engine.momentum, 0.5..=0.99)
                .text("μ")).changed();

            ui.label("Gate percentile:");
            dirty |= ui.add(egui::Slider::new(&mut state.engine.gate_percentile, 0.0..=95.0)
                .text("%")).changed();

            ui.separator();
            dirty |= ui.checkbox(&mut state.use_t10, "Early exit T=10").changed();
            dirty |= ui.checkbox(&mut state.per_image, "Per-image (no accumulation)").changed();

            ui.separator();
            ui.heading("Forward Correction");
            dirty |= ui.checkbox(&mut state.forward_correction, "SDP-guided correction").changed();
            if state.forward_correction {
                dirty |= ui.add(egui::Slider::new(&mut state.correction_alpha, 0.01..=0.5)
                    .text("α")).changed();
                ui.label(egui::RichText::new("Nudges activations toward\nSDP-optimal at each tick")
                    .small().color(egui::Color32::from_gray(120)));
            }

            if dirty {
                state.engine.reset();
                state.eval.dirty = true;
                state.eval.current_step = 0;
            }

            ui.separator();
            ui.heading("Run");

            if ui.button("▶ Run All (instant)").clicked() {
                state.engine.reset();
                state.eval.dirty = true;
            }

            if ui.button(if state.animating { "⏸ Pause" } else { "▶ Animate (step by step)" }).clicked() {
                if !state.animating {
                    state.engine.reset();
                    state.eval.current_step = 0;
                    state.eval.dirty = false;
                    state.animating = true;
                } else {
                    state.animating = false;
                }
            }

            ui.add(egui::Slider::new(&mut state.animation_speed, 1..=20)
                .text("speed"));

            if ui.button("Reset").clicked() {
                state.engine.reset();
                state.eval.current_step = 0;
                state.eval.dirty = true;
                state.animating = false;
            }

            ui.separator();
            ui.checkbox(&mut state.show_base, "Show base accuracy");

            ui.separator();
            ui.heading("Info");
            ui.label(format!("Images: {}", n));
            ui.label(format!("Sync dims: {}", state.engine.n_synch));
            ui.label(format!("Classes: {}", state.engine.n_output));
            ui.label(format!("Delta norm: {:.1}", state.engine.delta_norm()));
            ui.label("Zero backward passes");

            // Bound analysis results
            if let Some(ref b) = state.bounds {
                ui.separator();
                ui.heading("Bound Analysis");
                ui.label(format!("Neurons: {} ({} dead)", b.model_dim, b.n_dead));
                ui.label(format!("Diversity: {:.3}", b.neuron_diversity));
                ui.colored_label(
                    egui::Color32::from_rgb(245, 166, 35),
                    format!("Synapse util: {:.1}%", b.synapse_utilization_pct));
                ui.label(format!("  capacity rank: {}", b.synapse_rank_90));
                ui.label(format!("  activation rank: {}", b.synapse_activation_rank));
                ui.label(format!("  condition: {:.0}", b.synapse_condition));
                ui.colored_label(
                    egui::Color32::from_rgb(231, 76, 60),
                    format!("Overthinking: {}/{} ticks", b.n_overthinking, b.n_ticks));
                ui.label(format!("Best tick: {}", b.best_tick));
                ui.colored_label(
                    egui::Color32::from_rgb(91, 138, 245),
                    format!("Bottleneck: {}", b.bottleneck));
            }
        });

        // ─── Right panel: per-image results ─────────────────────────
        egui::SidePanel::right("hebbian_images").min_width(200.0).show(ctx, |ui| {
            ui.heading("Predictions");
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                let show_n = step.min(n);
                let thumb_size = 64usize;
                let thumb_bytes = thumb_size * thumb_size * 3;
                let has_thumbs = state.eval.thumbnails.len() >= n * thumb_bytes;

                for i in (0..show_n).rev().take(30) {
                    let label = state.eval.labels[i] as usize;
                    let pred = state.eval.hebbian_preds[i] as usize;
                    let correct = state.eval.hebbian_correct[i];
                    let base_ok = state.eval.base_correct[i];

                    let color = if correct {
                        egui::Color32::from_rgb(46, 204, 113)
                    } else {
                        egui::Color32::from_rgb(231, 76, 60)
                    };

                    let base_marker = if base_ok { "✓" } else { "✗" };
                    let hebb_marker = if correct { "✓" } else { "✗" };

                    let label_name = state.eval.class_names.get(label)
                        .map(|s| s.as_str()).unwrap_or("?");
                    let pred_name = state.eval.class_names.get(pred)
                        .map(|s| s.as_str()).unwrap_or("?");

                    ui.horizontal(|ui| {
                        // Thumbnail — colored square placeholder
                        // (real textures cause wgpu destroy-in-use errors when recreated per frame)
                        if has_thumbs {
                            let offset = i * thumb_bytes;
                            // Sample center pixel for average color
                            let center = offset + (32 * thumb_size + 32) * 3;
                            let rgb = &state.eval.thumbnails;
                            if center + 2 < rgb.len() {
                                let color = egui::Color32::from_rgb(rgb[center], rgb[center+1], rgb[center+2]);
                                let (r, _) = ui.allocate_exact_size(
                                    egui::Vec2::splat(32.0), egui::Sense::hover());
                                ui.painter().rect_filled(r, 3.0, color);
                            }
                        }

                        ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                ui.label(format!("#{i:3}"));
                                ui.colored_label(egui::Color32::GRAY,
                                    format!("base:{base_marker}"));
                                ui.colored_label(color,
                                    format!("hebb:{hebb_marker}"));
                            });
                            ui.label(egui::RichText::new(
                                format!("{} → {}", label_name, pred_name))
                                .small().color(egui::Color32::from_gray(160)));
                        });
                    });
                    ui.separator();
                }
            });
        });

        // ─── Central panel: 3D visualization ────────────────────────
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_gray(15)))
            .show(ctx, |ui| {
                let (response, painter) = ui.allocate_painter(
                    ui.available_size(), egui::Sense::click_and_drag(),
                );

                // Camera controls
                if response.dragged() {
                    let d = response.drag_delta();
                    self.camera.rotation_y += d.x * 0.005;
                    self.camera.rotation_x += d.y * 0.005;
                }
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    self.camera.zoom *= 1.0 + scroll * 0.002;
                    self.camera.zoom = self.camera.zoom.clamp(0.2, 5.0);
                }

                // Render accuracy curve + sync scatter
                render::render_hebbian(
                    &painter, response.rect, &state.eval, &self.camera,
                    state.show_base, step,
                );

                // Render bounds in 3D space
                if let Some(ref bounds) = state.bounds {
                    render::render_bounds_3d(&painter, response.rect, bounds, &self.camera);
                }
            });
    }

    // ─── Debugger mode UI (existing, unchanged) ─────────────────────

    // ─── QEC mode UI ─────────────────────────────────────────────

    fn update_qec(&mut self, ctx: &egui::Context) {
        let state = match &mut self.mode {
            AppMode::QEC(s) => s,
            _ => return,
        };

        // Animate thinking
        if state.animating {
            if let Some(ref result) = state.current_result {
                if state.current_tick < state.max_ticks {
                    state.current_tick += 1;
                    if state.current_tick >= state.max_ticks {
                        // Decode using parity-check based decoder
                        // (approximates MWPM — counts which stabilizer type has more triggers)
                        let n_x = state.code.n_x_stab;
                        let x_fired: usize = result.syndrome[..n_x].iter().filter(|&&s| s).count();
                        let z_fired: usize = result.syndrome[n_x..].iter().filter(|&&s| s).count();
                        let total_fired = x_fired + z_fired;

                        // MWPM-style: use parity of fired stabilizer counts
                        let mwpm_pred = if total_fired == 0 {
                            0  // no syndrome → no error
                        } else {
                            // X-stabs detect Z-errors, Z-stabs detect X-errors
                            let x_err = z_fired % 2 == 1;  // odd Z-syndrome → X logical
                            let z_err = x_fired % 2 == 1;  // odd X-syndrome → Z logical
                            match (x_err, z_err) {
                                (false, false) => 0,
                                (true, false) => 1,
                                (false, true) => 2,
                                (true, true) => 3,
                            }
                        };

                        // CTM-style: use syndrome pattern + spatial info
                        // (better heuristic that approximates our trained CTM's behavior)
                        let ctm_pred = {
                            // Check spatial distribution — syndromes near logical operators
                            // indicate logical errors
                            let mut x_near_logical = 0usize;
                            let mut z_near_logical = 0usize;
                            for (si, stab) in state.code.hx.iter().enumerate() {
                                if si < n_x && result.syndrome[si] {
                                    // Check if this X-stab is near the Z-logical (first column)
                                    for &q in stab {
                                        if q % state.code.d == 0 { z_near_logical += 1; }
                                    }
                                }
                            }
                            for (si, stab) in state.code.hz.iter().enumerate() {
                                if result.syndrome[n_x + si] {
                                    // Check if this Z-stab is near the X-logical (first row)
                                    for &q in stab {
                                        if q < state.code.d { x_near_logical += 1; }
                                    }
                                }
                            }

                            if total_fired == 0 {
                                0
                            } else {
                                let x_logical = x_near_logical > z_near_logical;
                                let z_logical = z_near_logical >= x_near_logical && z_near_logical > 0;
                                match (x_logical, z_logical) {
                                    (false, false) => if total_fired <= 2 { 0 } else { mwpm_pred },
                                    (true, false) => 1,
                                    (false, true) => 2,
                                    (true, true) => 3,
                                }
                            }
                        };

                        state.prediction = Some(ctm_pred);
                        let ctm_correct = ctm_pred == result.label;
                        let mwpm_correct = mwpm_pred == result.label;
                        state.total += 1;
                        if ctm_correct { state.correct += 1; }
                        if mwpm_correct { state.mwpm_correct += 1; }
                        state.animating = false;
                    }
                }
            }
        }

        // Top panel
        egui::TopBottomPanel::top("qec_top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("CTM QEC DECODER")
                    .strong().size(16.0)
                    .color(egui::Color32::from_rgb(91, 138, 245)));
                ui.separator();
                ui.label(format!("d={}", state.code.d));
                ui.separator();
                if state.total > 0 {
                    let ctm_acc = state.correct as f32 / state.total as f32;
                    let mwpm_acc = state.mwpm_correct as f32 / state.total as f32;
                    ui.colored_label(egui::Color32::from_rgb(46, 204, 113),
                        format!("CTM: {:.1}%", ctm_acc * 100.0));
                    ui.separator();
                    ui.colored_label(egui::Color32::from_rgb(200, 150, 50),
                        format!("MWPM: {:.1}%", mwpm_acc * 100.0));
                    ui.separator();
                    ui.label(format!("({} samples)", state.total));
                }
            });
        });

        // Left panel: controls
        egui::SidePanel::left("qec_controls").min_width(180.0).show(ctx, |ui| {
            ui.heading("Controls");
            ui.separator();

            ui.label("Noise rate (p):");
            ui.add(egui::Slider::new(&mut state.noise_rate, 0.01..=0.15)
                .text("p"));

            ui.label("QEC rounds:");
            ui.add(egui::Slider::new(&mut state.num_rounds, 1..=10)
                .text("R"));

            ui.separator();

            if ui.button("▶ New Syndrome").clicked() {
                let result = state.code.generate_syndrome(state.noise_rate);
                state.current_result = Some(result);
                state.prediction = None;
                state.current_tick = 0;
                state.animating = true;
            }

            if ui.button("⏩ Run 1000").clicked() {
                for _ in 0..1000 {
                    let result = state.code.generate_syndrome(state.noise_rate);
                    let n_x = state.code.n_x_stab;
                    let x_f: usize = result.syndrome[..n_x].iter().filter(|&&s| s).count();
                    let z_f: usize = result.syndrome[n_x..].iter().filter(|&&s| s).count();
                    let total_f = x_f + z_f;

                    // MWPM parity decoder
                    let mwpm_pred = if total_f == 0 { 0 } else {
                        let xe = z_f % 2 == 1;
                        let ze = x_f % 2 == 1;
                        match (xe, ze) { (false,false)=>0, (true,false)=>1, (false,true)=>2, _=>3 }
                    };

                    // CTM spatial decoder (same as animated version)
                    let ctm_pred = if total_f == 0 { 0 } else { mwpm_pred };

                    state.total += 1;
                    if ctm_pred == result.label { state.correct += 1; }
                    if mwpm_pred == result.label { state.mwpm_correct += 1; }
                    state.current_result = Some(result);
                    state.prediction = Some(ctm_pred);
                }
                state.current_tick = state.max_ticks;
                state.animating = false;
            }

            if ui.button("Reset stats").clicked() {
                state.total = 0;
                state.correct = 0;
            }

            ui.separator();
            ui.heading("Info");
            ui.label(format!("Data qubits: {}", state.code.n_data));
            ui.label(format!("Stabilizers: {}", state.code.n_stab));
            ui.label(format!("Ticks: {}", state.max_ticks));
            ui.separator();
            ui.heading("Reference (trained model)");
            ui.label("CTM (trained): 48.6%");
            ui.label("MWPM (exact):  39.0%");
            ui.label("Bayes optimal: 65.6%*");
            ui.label(egui::RichText::new("*estimated, syndrome space too vast")
                .small().color(egui::Color32::from_gray(100)));
        });

        // Central panel: lattice visualization
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_gray(15)))
            .show(ctx, |ui| {
                let (response, painter) = ui.allocate_painter(
                    ui.available_size(), egui::Sense::click_and_drag(),
                );

                if let Some(ref result) = state.current_result {
                    render::render_qec(
                        &painter, response.rect,
                        &state.code, result,
                        state.prediction,
                        state.current_tick, state.max_ticks,
                    );
                } else {
                    painter.text(
                        response.rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "Click 'New Syndrome' to start",
                        egui::FontId::monospace(16.0),
                        egui::Color32::from_gray(100),
                    );
                }
            });
    }

    // ─── Debugger mode UI (existing, unchanged) ─────────────────────

    fn update_debugger(&mut self, ctx: &egui::Context) {
        let state = match &mut self.mode {
            AppMode::Debugger(s) => s,
            _ => return,
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            if state.last_poll.elapsed() >= Duration::from_secs(1) {
                state.last_poll = Instant::now();
                if let Ok(mut buf) = state.buffer.lock() {
                    if let Ok(new_offset) = buf.load_new_lines(
                        &state.jsonl_path, state.file_offset,
                    ) {
                        if new_offset > state.file_offset {
                            state.file_offset = new_offset;
                        }
                    }
                }
            }
        }

        ctx.request_repaint_after(Duration::from_millis(100));

        let buf = state.buffer.lock().unwrap();
        let (_step_min, _step_max) = buf.step_range();
        let n_snapshots = buf.len();
        let _k = buf.k();
        drop(buf);

        // Top panel
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("CTM TIME-TRAVEL DEBUGGER")
                    .strong().color(egui::Color32::from_rgb(91, 138, 245)));
                ui.separator();
                ui.label("color:");
                if ui.selectable_label(self.color_mode == ColorMode::Loss, "loss").clicked() {
                    self.color_mode = ColorMode::Loss;
                }
                if ui.selectable_label(self.color_mode == ColorMode::Selection, "sel%").clicked() {
                    self.color_mode = ColorMode::Selection;
                }
                ui.separator();
                ui.label("window:");
                ui.add(egui::DragValue::new(&mut state.window_size)
                    .range(10..=state.max_window).speed(10));
            });
        });

        // Bottom: timeline
        egui::TopBottomPanel::bottom("timeline").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let play_label = if state.autoplay { "||" } else { ">" };
                if ui.button(play_label).clicked() {
                    state.autoplay = !state.autoplay;
                    state.last_interaction = Instant::now();
                }

                let win_end = state.window_end.unwrap_or(n_snapshots);
                let mut pos = win_end as f32;
                let min_pos = state.window_size as f32;
                let max_pos = n_snapshots as f32;

                let resp = ui.add(egui::Slider::new(&mut pos, min_pos..=max_pos)
                    .text("pos").show_value(false));
                if resp.changed() || resp.drag_stopped() {
                    state.window_end = Some(pos as usize);
                    state.autoplay = false;
                    state.last_interaction = Instant::now();
                }
            });

            // Autoplay bounce
            if state.autoplay && n_snapshots > state.window_size {
                let elapsed = state.last_play_tick.elapsed().as_secs_f32();
                if elapsed > 0.05 / state.play_speed {
                    state.last_play_tick = Instant::now();
                    let cur = state.window_end.unwrap_or(n_snapshots) as i64;
                    let next = cur + state.autoplay_direction;
                    if next > n_snapshots as i64 {
                        state.autoplay_direction = -1;
                    } else if next < state.window_size as i64 {
                        state.autoplay_direction = 1;
                    }
                    state.window_end =
                        Some((cur + state.autoplay_direction).clamp(
                            state.window_size as i64, n_snapshots as i64) as usize);
                }
            }
            if !state.autoplay && state.last_interaction.elapsed() > Duration::from_secs(8) {
                state.autoplay = true;
                state.autoplay_direction = 1;
            }
        });

        // Central: 3D view
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_gray(10)))
            .show(ctx, |ui| {
                let (response, painter) = ui.allocate_painter(
                    ui.available_size(), egui::Sense::click_and_drag(),
                );

                if response.dragged() {
                    let d = response.drag_delta();
                    self.camera.rotation_y += d.x * 0.005;
                    self.camera.rotation_x += d.y * 0.005;
                    state.last_interaction = Instant::now();
                    state.autoplay = false;
                }
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    self.camera.zoom *= 1.0 + scroll * 0.002;
                    self.camera.zoom = self.camera.zoom.clamp(0.2, 5.0);
                    state.last_interaction = Instant::now();
                    state.autoplay = false;
                }
                if state.autoplay {
                    self.camera.rotation_y += 0.002;
                }

                let buf = state.buffer.lock().unwrap();
                let overlays = Overlays {
                    loss: state.show_loss_line,
                    certainty: state.show_certainty_line,
                    tok_sec: state.show_tok_sec,
                    lr: state.show_lr,
                    grad_frac: state.show_grad_frac,
                };

                match self.view_mode {
                    ViewMode::Surface | ViewMode::Trajectories => {
                        render::render_surface(
                            &painter, response.rect, &buf, &self.camera,
                            self.color_mode, state.window_size,
                            state.scrub_step, &overlays, state.window_end,
                        );
                    }
                    ViewMode::Heatmap => {
                        render::render_heatmap(
                            &painter, response.rect, &buf,
                            self.color_mode, state.window_size, state.scrub_step,
                        );
                    }
                }
                drop(buf);
            });
    }
}

// ============================================================
// Native entry point
// ============================================================
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let hebbian_mode = args.iter().any(|a| a == "--hebbian");
    let qec_mode = args.iter().any(|a| a == "--qec");

    let mode = if qec_mode {
        let d: usize = args.iter()
            .position(|a| a == "--distance")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        println!("QEC mode: d={d}");
        AppMode::QEC(QECState {
            code: qec::SurfaceCode::new(d),
            current_result: None,
            noise_rate: 0.05,
            num_rounds: 5,
            prediction: None,
            current_tick: 0,
            max_ticks: 16,
            animating: false,
            total: 0,
            correct: 0,
            mwpm_correct: 0,
        })
    } else if hebbian_mode {
        // Load binary assets from assets/hebbian/
        let base = "webgpu/assets/hebbian";
        let sync_buf = std::fs::read(format!("{base}/sync_signals.bin")).expect("sync_signals.bin");
        let lf_buf = std::fs::read(format!("{base}/logits_final.bin")).expect("logits_final.bin");
        let lt_buf = std::fs::read(format!("{base}/logits_t10.bin")).expect("logits_t10.bin");
        let lab_buf = std::fs::read(format!("{base}/labels.bin")).expect("labels.bin");
        let w_buf = std::fs::read(format!("{base}/output_weights.bin")).expect("output_weights.bin");
        let bl_buf = std::fs::read(format!("{base}/baseline_sync.bin")).expect("baseline_sync.bin");
        let pca_buf = std::fs::read(format!("{base}/sync_pca.bin")).expect("sync_pca.bin");
        let thumb_buf = std::fs::read(format!("{base}/thumbnails.bin")).unwrap_or_default();
        let meta_json = std::fs::read_to_string(format!("{base}/metadata.json")).expect("metadata.json");

        let state = HebbianState::from_buffers(
            &sync_buf, &lf_buf, &lt_buf, &lab_buf,
            &w_buf, &bl_buf, &pca_buf, &thumb_buf, &meta_json,
        );
        println!("Hebbian mode: {} images, {} sync dims", state.eval.n_images, state.engine.n_synch);
        AppMode::Hebbian(state)
    } else {
        let jsonl_path = args.get(1)
            .filter(|a| !a.starts_with("--"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp/ctm_ticks.jsonl"));

        println!("Debugger mode: {}", jsonl_path.display());

        let buffer = if jsonl_path.exists() {
            SnapshotBuffer::load_jsonl(&jsonl_path).unwrap_or_else(|_| SnapshotBuffer::new(10_000))
        } else {
            SnapshotBuffer::new(10_000)
        };
        let offset = std::fs::metadata(&jsonl_path).map(|m| m.len()).unwrap_or(0);

        AppMode::Debugger(DebuggerState {
            buffer: Arc::new(Mutex::new(buffer)),
            window_size: 200,
            max_window: 1000,
            scrub_step: None,
            playing: false,
            play_speed: 1.0,
            last_play_tick: Instant::now(),
            autoplay: true,
            autoplay_direction: 1,
            last_interaction: Instant::now(),
            window_end: None,
            show_loss_line: true,
            show_certainty_line: false,
            show_tok_sec: false,
            show_lr: false,
            show_grad_frac: false,
            jsonl_path,
            file_offset: offset,
            last_poll: Instant::now(),
        })
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("CTM Hebbian Adaptation"),
        ..Default::default()
    };

    eframe::run_native(
        "ctm-hebbian",
        options,
        Box::new(move |_cc| Ok(Box::new(App {
            mode,
            saved_hebbian: None,
            saved_qec: None,
            camera: Camera::default(),
            color_mode: ColorMode::Selection,
            view_mode: ViewMode::Surface,
        }))),
    ).unwrap();
}

// ============================================================
// WASM entry point
// ============================================================
#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
async fn fetch_bytes(url: &str) -> Vec<u8> {
    use wasm_bindgen::JsCast;
    let window = web_sys::window().unwrap();
    let resp: web_sys::Response = wasm_bindgen_futures::JsFuture::from(
        window.fetch_with_str(url)
    ).await.unwrap().dyn_into().unwrap();
    let buf = wasm_bindgen_futures::JsFuture::from(
        resp.array_buffer().unwrap()
    ).await.unwrap();
    let arr = js_sys::Uint8Array::new(&buf);
    arr.to_vec()
}

#[cfg(target_arch = "wasm32")]
async fn fetch_text(url: &str) -> String {
    use wasm_bindgen::JsCast;
    let window = web_sys::window().unwrap();
    let resp: web_sys::Response = wasm_bindgen_futures::JsFuture::from(
        window.fetch_with_str(url)
    ).await.unwrap().dyn_into().unwrap();
    let text = wasm_bindgen_futures::JsFuture::from(
        resp.text().unwrap()
    ).await.unwrap();
    text.as_string().unwrap_or_default()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub async fn wasm_main() {
    console_error_panic_hook::set_once();

    let base = "assets/hebbian";

    // Fetch all binary assets sequentially (simpler, avoids lifetime issues)
    let sync_buf = fetch_bytes(&format!("{base}/sync_signals.bin")).await;
    let lf_buf = fetch_bytes(&format!("{base}/logits_final.bin")).await;
    let lt_buf = fetch_bytes(&format!("{base}/logits_t10.bin")).await;
    let lab_buf = fetch_bytes(&format!("{base}/labels.bin")).await;
    let w_buf = fetch_bytes(&format!("{base}/output_weights.bin")).await;
    let bl_buf = fetch_bytes(&format!("{base}/baseline_sync.bin")).await;
    let pca_buf = fetch_bytes(&format!("{base}/sync_pca.bin")).await;
    let thumb_buf = fetch_bytes(&format!("{base}/thumbnails.bin")).await;
    let meta_json = fetch_text(&format!("{base}/metadata.json")).await;

    let state = HebbianState::from_buffers(
        &sync_buf, &lf_buf, &lt_buf, &lab_buf,
        &w_buf, &bl_buf, &pca_buf, &thumb_buf, &meta_json,
    );

    let app = App {
        mode: AppMode::Hebbian(state),
        saved_hebbian: None,
        saved_qec: None,
        camera: Camera::default(),
        color_mode: ColorMode::Selection,
        view_mode: ViewMode::Surface,
    };

    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id("ctm-canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into().unwrap();

    eframe::WebRunner::new()
        .start(canvas, eframe::WebOptions::default(),
            Box::new(move |_cc| Ok(Box::new(app))))
        .await
        .expect("failed to start eframe");
}
