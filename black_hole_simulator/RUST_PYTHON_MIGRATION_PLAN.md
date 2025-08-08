# Black Hole Simulator — Rust + Python Migration Plan

This plan outlines a two-phase migration:
- Phase 1: CPU Rust core exposed to Python (viewer/UI)
- Phase 2: GPU acceleration in Rust using wgpu + WGSL, keeping the Python viewer

The goal is to achieve correctness and a smooth UX quickly (Phase 1), then unlock performance and portability (Phase 2).

---

## Repository Structure (target)

```
black_hole_simulator/
  CODEBASE_OVERVIEW.md
  RUST_PYTHON_MIGRATION_PLAN.md
  rust_core/
    Cargo.toml
    src/
      lib.rs
      geodesic.rs
      camera.rs
      scene.rs
      math.rs
      cpu_renderer.rs
      gpu/               # Phase 2
        mod.rs
        pipelines.rs
        shader.wgsl
    tests/
      geodesic_tests.rs
  python_viewer/
    pyproject.toml       # if using maturin for mixed tree, otherwise own venv
    app/
      main.py
      ui/
        window.py
        controls.py
      core/
        bridge.py        # Rust FFI wrapper via pyo3
        types.py
      assets/
        icons/
  cpp_legacy/            # optional: keep C++ refs or test scenes
    black_hole.cpp
    geodesic.comp
    CPU-geodesic.cpp
    2D_lensing.cpp
    ray_tracing.cpp
```

---

## Phase 1 — Rust CPU Core + Python Viewer

Objective: Port null-geodesic RK4 integration and CPU rendering path from `CPU-geodesic.cpp` to Rust; expose via Python (pyo3) and display frames in a Python GUI.

### Deliverables
- Rust crate `rust_core` exporting:
  - `render_frame(params: RenderParams) -> Py<PyBytes>` returning RGBA bytes
  - `set_scene(scene: SceneDesc)` and `set_camera(cam: CameraDesc)`
  - Deterministic CPU geodesic integrator (`geodesicRHS`, `rk4_step`)
- Python app `python_viewer` (PySide6 or PyQt6) showing frames at interactive rate with controls for camera, disk, objects, steps, and D_LAMBDA.
- Tests validating trajectories against the C++ CPU reference for known seeds.

### Tasks
- __[math]__ Choose Rust math crate (`glam` or `nalgebra`); implement `Vec3`, `mat` utils if needed.
- __[core]__ Port structs `Camera`, `Ray`, `BlackHole`, `Object`, `Scene` mirroring the CPU design.
- __[integrator]__ Translate `geodesicRHS(...)` and `rk4Step(...)` to Rust (`geodesic.rs`).
- __[renderer]__ CPU renderer that, per pixel, constructs camera ray and marches geodesic until hit/escape; writes RGBA into `Vec<u8>`.
- __[ffi]__ Add `pyo3` interface in `lib.rs`:
  - Convert Python dicts/objects to Rust structs (params, scene, camera).
  - Return `PyBytes` of the RGBA frame.
- __[packaging]__ `maturin` config to build a Python wheel; dev workflow docs.
- __[python-ui]__ PySide6 UI:
  - QMainWindow with QLabel/QImage for frame, timer-driven updates.
  - Controls: Orbit (mouse), zoom (wheel), sliders for steps/resolution, disk radii; object list.
  - Bridge to Rust: `bridge.py` wraps the FFI calls; marshals types.
- __[tests]__ Unit tests for integrator (single-step, multi-step), hit tests vs. BH and objects. Optionally, image snapshot tests at low res.
- __[perf]__ Add dynamic quality:
  - Lower resolution and/or steps when camera is moving.
  - Configurable `WIDTH/HEIGHT`, `STEPS`, `D_LAMBDA` from Python.

### Acceptance Criteria
- 200×150 render at interactive rates on CPU (≥5–10 FPS) with reduced settings while moving.
- Image features: BH silhouette, basic disk proxy (CPU version may initially omit the disk plane test), and object hits.
- Cross-check one or two trajectories with legacy CPU code within tolerance.

### Time Estimate
- 1–2 weeks (solo), depending on UI polish and test depth.

---

## Phase 2 — Rust GPU with wgpu + WGSL (Python Viewer retained)

Objective: Replace CPU marching with a GPU compute pipeline using wgpu, porting the GLSL compute shader (`geodesic.comp`) to WGSL and maintaining Python integration.

### Deliverables
- wgpu compute pipeline:
  - Bind groups mirroring `Camera`, `Disk`, `Objects` buffers.
  - WGSL shader port of geodesic integrator.
  - Output texture -> mapped to CPU for Python display (or zero-copy path if later integrated with a Rust window).
- Rust FFI unchanged at the Python boundary (`render_frame(params) -> bytes`), now backed by GPU.
- Optional: Grid overlay as a separate compute/draw pass; or provide as a toggle.

### Tasks
- __[wgpu setup]__ Initialize instance/device/queue; choose adapter; create compute pipeline.
- __[buffers]__ Define Rust structs with `#[repr(C)]` mirroring uniform/storage layouts:
  - `CameraUBO`, `DiskUBO`, `ObjectsSSBO` (std430 style); alignment rules per WGSL.
- __[shader port]__ Translate `geodesic.comp` GLSL to WGSL:
  - Replace UBOs with WGSL `@group/@binding` uniform/storage buffers.
  - Replace `imageStore` with writing to a storage texture or a storage buffer representing the image.
  - Reimplement `rk4Step`, `geodesicRHS` in WGSL.
  - Keep constants (`D_LAMBDA`, steps) configurable via push constants or uniforms.
- __[dispatch]__ Compute grid and workgroup sizes matching resolution; handle odd sizes.
- __[readback]__ Resolve storage texture to a mapped staging buffer and copy to `Vec<u8>` for Python.
- __[perf]__ Implement dynamic quality while moving (reduced resolution/steps); measure GPU time.
- __[validation]__ Compare GPU vs CPU outputs on test seeds; assert bounded error.

### Acceptance Criteria
- GPU renders match CPU within expected tolerance on test scenes.
- 800×600 at smooth interactive rates on a mid-tier GPU with dynamic quality scaling.
- Python viewer unchanged from a user perspective.

### Time Estimate
- 5–8 weeks (solo), depending on shader-port complexity and readback integration.

---

## Milestones & Timeline (indicative)

1. __M1 (Week 1)__: Rust CPU integrator + minimal FFI; Python displays first image.
2. __M2 (Week 2)__: Interactive controls, dynamic quality; tests pass; packaged wheel.
3. __M3 (Week 4)__: wgpu device + buffers + WGSL shader skeleton; first compute dispatch.
4. __M4 (Week 6)__: Full integrator ported; correct hits (BH/disk/objects); stable readback.
5. __M5 (Week 8)__: Performance tuning, grid overlay option, documentation.

---

## Risks & Mitigations

- __GPU readback overhead__: Use smaller resolution while moving; batch copies; consider persistent staging buffers.
- __WGSL port divergences__: Build parity tests; start from simple cases (no disk/objects), then add features.
- __ABI/FFI friction__: Keep FFI surface minimal; use `pyo3` types and `maturin` for reliable packaging.
- __Precision differences__: Validate with controlled seeds and step sizes; document tolerances.

---

## Tools & Dependencies

- Rust: `pyo3`, `maturin`, `wgpu`, `bytemuck`, `glam`/`nalgebra`, `anyhow`, `thiserror`, `serde`.
- Python: `PySide6`/`PyQt6`, `numpy` (optional for buffer handling), `mypy` (optional), `pytest`.

---

## Next Steps

- Confirm crate structure and math library choice (`glam` vs `nalgebra`).
- Initialize `rust_core` with `pyo3` and a stub `render_frame` returning a solid color.
- Scaffold `python_viewer` window that can display raw RGBA buffers.
- Start porting `geodesicRHS` and `rk4_step` from `CPU-geodesic.cpp`.
- Define parameter schemas (`RenderParams`, `CameraDesc`, `SceneDesc`) shared across Rust/Python.
