# Black Hole Simulator — Codebase Overview

This document explains the structure and functionality of the repository, summarizing each major component and how they interact. It is based on a review of the files present under `black_hole_simulator/`.

## Repository Structure

- `README.md`
- `black_hole.cpp`
- `geodesic.comp`
- `CPU-geodesic.cpp`
- `2D_lensing.cpp`
- `ray_tracing.cpp`
- `black_hole.exe` (prebuilt Windows binary)
- `Gravity_Sim.zip` (archive)
- `vs_code/`
  - `c_cpp_properties.json`
  - `launch.json`
  - `settings.json`
  - `tasks.json`

## High-Level Purpose

The repository contains several OpenGL-based programs that visualize gravitational effects near a black hole, escalating from a simple 2D lensing demo to a GPU-accelerated 3D geodesic ray tracer:

- 2D null-geodesic visualization with trails (`2D_lensing.cpp`).
- Classical CPU ray tracer demo with spheres, shading, and shadows (`ray_tracing.cpp`).
- CPU-based 3D geodesic ray tracing with optional full null-geodesic integration (`CPU-geodesic.cpp`).
- GPU compute-shader geodesic ray tracing with support for an accretion disk and scene objects (`black_hole.cpp` + `geodesic.comp`).

Dependencies across programs:
- OpenGL, GLFW (windowing/context), GLEW (function loading)
- GLM (math)
- C++ standard library

## Quick Start Notes (from `README.md`)

- 2D: compile and run `2D_lensing.cpp` with required dependencies.
- 3D: `black_hole.cpp` works with `geodesic.comp` (GPU compute shader) to accelerate physics and rendering.
- The README references Windows + MSYS2 for installing GLFW/GLEW.
- Video link explains the code in detail: https://www.youtube.com/watch?v=8-B6ryuBkCM

## Core Concepts and Structures

- `BlackHole` (in multiple files):
  - Stores position, mass, and Schwarzschild radius `r_s = 2GM/c^2`.
  - `Intercept(...)` checks if a ray enters the event horizon.

- `Camera`:
  - `black_hole.cpp`: Orbit-only camera always centered on the black hole at the origin; left-drag orbits, scroll zooms; panning is disabled to keep focus on the BH.
  - `CPU-geodesic.cpp`: Orbit + pan (Shift+drag) + zoom; constructs camera basis vectors for ray generation.

- `Ray` (variants):
  - 3D CPU (`CPU-geodesic.cpp`): Spherical coords `(r, theta, phi)` and derivatives `(dr, dtheta, dphi)`, with conserved quantities `E` (energy) and `L` (angular momentum). Derived from initial position and direction.
  - 2D (`2D_lensing.cpp`): Reduced to `(r, phi)` and `(dr, dphi)` with a trail of points for visualization.
  - GPU (`geodesic.comp`): GLSL struct mirrored from the CPU design.

- Geodesic integrators (Runge–Kutta 4):
  - 3D CPU (`CPU-geodesic.cpp`): `geodesicRHS(...)` + `rk4Step(...)` integrate null geodesics in Schwarzschild spacetime.
  - 2D (`2D_lensing.cpp`): Simplified equations for planar case.
  - GPU (`geodesic.comp`): GLSL implementations `geodesicRHS(...)` and `rk4Step(...)`, executed per-pixel in the compute shader.

## File-by-File Summary

- `README.md`
  - Project goals: ray tracing, accretion disk, spacetime curvature demonstration, optional realtime.
  - How to run 2D and 3D variants, dependency notes.

- `black_hole.cpp` (GPU main)
  - Creates an OpenGL 4.3 core context and initializes GLEW.
  - Renders a fullscreen quad textured with output from a compute shader.
  - Sets up Uniform Buffer Objects (UBOs):
    - `cameraUBO` (binding=1): position, basis vectors, FOV, aspect, and a `moving` flag.
    - `diskUBO` (binding=2): accretion disk parameters (`disk_r1`, `disk_r2`, `disk_num`, `thickness`).
    - `objectsUBO` (binding=3): up to 16 objects, each with `posRadius` (xyz + radius), `color`, and `mass`.
  - Generates a “warped grid” mesh to visualize spacetime distortion via `generateGrid(...)` and draws it with `drawGrid(...)` using a separate grid shader program.
  - Important note: References `grid.vert` and `grid.frag` via `CreateShaderProgram("grid.vert", "grid.frag")`, but those files are not present in the repo. Grid won’t render unless added or the code is guarded.

- `geodesic.comp` (GLSL compute shader)
  - Layouts:
    - `image2D outImage` (binding=0) for writing final pixels.
    - UBO `Camera` (binding=1) matching CPU camera data.
    - UBO `Disk` (binding=2) for accretion disk parameters.
    - UBO `Objects` (binding=3) for scene objects (positions/radii/colors/masses).
  - Per-invocation steps:
    - Reconstructs a ray through the pixel using camera basis and FOV.
    - Initializes geodesic state `(r, theta, phi, dr, dtheta, dphi, E, L)` from position/direction.
    - Integrates with RK4 for many steps (`D_LAMBDA=1e7`, `steps=60000`).
    - Checks, in order:
      - Event horizon hit (`intercept(...)`): black color.
      - Accretion disk hit by crossing equatorial plane within `[disk_r1, disk_r2]`: disk-shaded color.
      - Object hit: simple view-dependent diffuse shading using object color and normal.
      - Escape to infinity: remain background.

- `CPU-geodesic.cpp` (CPU renderer)
  - Builds a window and fullscreen quad, renders pixel buffer via CPU ray tracing to a texture.
  - Toggles with `G` key:
    - Off: analytical ray-sphere hit test against the event horizon (red on hit).
    - On: full null-geodesic RK4 marching until hit or escape.
  - Camera offers orbit/pan/zoom; constructs `forward/right/up` for ray directions.

- `2D_lensing.cpp` (2D geodesics)
  - Immediate-mode GL to draw a red circle for the BH (`r_s` radius) and ray trails.
  - `Ray.step(...)` integrates 2D null geodesics, and `Ray.draw(...)` shows current points and fading trails.
  - Engine sets orthographic projection; includes state for pan/zoom offsets.

- `ray_tracing.cpp` (classical CPU ray tracer)
  - Minimal scene with spheres (`Object`), Lambert shading, and shadow rays.
  - Renders pixel-by-pixel on CPU each frame; uploads to GL texture for display.

## GPU Pipeline (`black_hole.cpp` + `geodesic.comp`)

1. CPU (`black_hole.cpp`):
   - Initialize OpenGL context, compile display shaders, and create compute program from `geodesic.comp`.
   - Populate and bind UBOs with camera, disk, and object data.
   - Dispatch the compute shader over a fixed compute resolution (currently 200×150 in shader code) writing into `outImage`.
   - Render the resulting image to a fullscreen quad; optionally draw a warped grid overlay (requires missing grid shaders).

2. GPU (`geodesic.comp`):
   - For each pixel, construct a camera ray and initialize geodesic state.
   - Integrate the null geodesic via RK4, checking for black hole, accretion disk, or object hits.
   - Shade and write out the pixel color.

## Controls

- `black_hole.cpp`:
  - Left mouse: orbit around origin.
  - Mouse wheel: zoom (clamped between `minRadius` and `maxRadius`).
  - Right mouse: temporarily toggles a `Gravity` flag.
  - `G` key: toggles `Gravity` on/off with console message.
  - Camera always targets `(0,0,0)`.

- `CPU-geodesic.cpp`:
  - Left drag: orbit.
  - Shift + Left drag: pan.
  - Mouse wheel: zoom.
  - `G` key: toggle between simple hit test and full geodesic integration.

## Notable Details and Caveats

- Missing shaders: `grid.vert` and `grid.frag` are referenced by `black_hole.cpp` but not present. Grid overlay will fail unless these are added or calls are guarded.
- Fixed compute resolution: The compute shader currently uses constants `WIDTH=200`, `HEIGHT=150`. Scaling up requires dispatching more workgroups and ensuring the output image matches.
- Performance tuning: The shader integrates with `steps=60000` regardless of the camera `moving` flag; dynamic quality scaling (lower steps/res while moving) is ready in design but not yet leveraged in the shader.
- Units: Physical constants are SI (c, G). Scene positions/radii are in meters on astronomical scales. The Sagittarius A* BH mass is set as `8.54e36` kg.

## Build/Run (High Level)

- Install dependencies: GLFW, GLEW, GLM, and OpenGL driver support. (MSYS2 on Windows is suggested in the README.)
- Compile each `.cpp` to an executable, linking against OpenGL, GLEW, GLFW, and GLM.
- For the GPU version:
  - Ensure `geodesic.comp` is accessible at runtime (usually next to the executable or with proper paths).
  - Add or remove the grid overlay calls depending on availability of `grid.vert`/`grid.frag`.
  - The included `black_hole.exe` may work out-of-the-box depending on GPU/driver.

## Potential Improvements

- Add the missing grid shaders or conditionally disable grid rendering when they are absent.
- Increase compute resolution and implement dynamic resolution/step count when the camera is moving.
- Add more realistic accretion disk modeling and texturing; support multiple light sources.
- Expose parameters (object masses, disk radii, integration step size and count) via a UI or config file.
- Abstract shared math/structures to reduce duplication across CPU and GPU paths.

## References to Key Symbols

- `black_hole.cpp`:
  - Camera: `Camera::position()`, `Camera::processMouseMove()`, `Camera::processScroll()`, `Camera::processKey()`
  - UBOs: `cameraUBO` (binding 1), `diskUBO` (binding 2), `objectsUBO` (binding 3)
  - Grid: `Engine::generateGrid()`, `Engine::drawGrid()`

- `geodesic.comp`:
  - UBO layouts: `uniform Camera`, `uniform Disk`, `uniform Objects`
  - Integration: `geodesicRHS(...)`, `rk4Step(...)`, `intercept(...)`, `interceptObject(...)`, `crossesEquatorialPlane(...)`

- `CPU-geodesic.cpp`:
  - Geodesic integration: `geodesicRHS(...)`, `rk4Step(...)`
  - Ray tracing: `raytrace(...)`

- `2D_lensing.cpp`:
  - Integration: `rk4Step(...)`, `geodesicRHS(...)`
  - Visualization: `BlackHole::draw()`, `Ray::draw(...)`

---

Generated on 2025-08-08.
