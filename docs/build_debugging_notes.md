# Notes on Python-Rust Integration Debugging

This document summarizes the extensive debugging process undertaken to resolve build and import issues for the `aqft_curved` package. The core challenge was correctly packaging a mixed Python/Rust project using `maturin`.

## Initial State & Problem

- **Goal:** Build a Python package (`aqft_curved`) that uses a compiled Rust extension (`_core`) for performance-critical code.
- **Problem:** After building and installing the package, Python could not find the module, consistently raising `ModuleNotFoundError: No module named 'aqft_curved'` or `ImportError: cannot import name '_core' from 'aqft_curved'`. 

## Summary of Debugging Attempts

The resolution involved a multi-step process of eliminating potential causes.

### 1. Build Method: `develop` vs. `build`

- **Attempt:** Initially used `maturin develop` for an editable install.
- **Problem:** This caused Python to import the local `aqft_curved` source directory, which does not contain the compiled Rust extension, leading to import errors.
- **Solution:** Switched to `maturin build` to create a standard wheel (`.whl`) file and installed it with `pip`. This is the standard and more robust method for package distribution and testing.

### 2. Environment Consistency

- **Problem:** At several points, commands were being run with the system's global Python interpreter instead of the project's dedicated virtual environment (`.venv`). This meant `pip install` was placing the package in a location that the test script's interpreter couldn't see.
- **Solution:** Strictly enforced the use of the virtual environment's interpreter for all commands (e.g., `c:\...\.venv\Scripts\python.exe -m pip ...`). This ensured that the build, installation, and execution all occurred in the same, consistent environment.

### 3. Project Structure & Configuration (`pyproject.toml`)

This was the most complex and persistent issue, with several iterative refinements.

- **Initial Structure:** The project was split into `python/` and `rust/` subdirectories, with `pyproject.toml` located inside `python/`.
- **Problem:** `maturin` failed to automatically discover and package the Python source files from `python/aqft_curved`.

- **Attempt 3a: Restructuring the Project:**
  - Moved the `pyproject.toml` to the project root.
  - Moved the Python package source (`aqft_curved/`) to the project root.
  - Deleted the now-redundant `python/` directory.
  - **Result:** This aligned the project with a standard, conventional layout that build tools are designed to expect.

- **Attempt 3b: Explicit Configuration (`python-source`, `module`):**
  - We tried explicitly telling `maturin` where to find the code with keys like `python-source = "..."` and `module = "aqft_curved._core"`.
  - **Problem:** These settings, while seemingly correct, were conflicting with another, more fundamental setting, leading to incomplete package builds.

### 4. The Breakthrough: The `features` Key

- **Root Cause:** By inspecting the contents of the generated wheel file (`tar -tf ...`), we discovered it contained the Rust extension but was completely missing all Python source files.
- **The Culprit:** The `pyproject.toml` file contained the line `features = ["pyo3/extension-module"]`.
- **Explanation:** This feature flag is intended for projects that are **only** a Rust extension. It explicitly tells `maturin` to build an extension module and *ignore* any accompanying Python source code. This was the direct cause of our incomplete packages.
- **The Final Fix:** Removing the `features` key switched `maturin` to its default "mixed-project" build mode. In this mode, it correctly located both the `aqft_curved` Python package and the `_core` Rust extension, bundling them together into a single, complete, and valid wheel.

## Final Working Configuration

1.  **Project Structure:**
    ```
    aqft_py/
    ├── aqft_curved/          # Python package source
    │   ├── __init__.py
    │   └── ...
    ├── docs/
    ├── rust/                 # Rust crate source
    │   ├── Cargo.toml
    │   └── src/
    ├── .venv/
    ├── pyproject.toml        # Main build config at ROOT
    └── ...
    ```

2.  **`pyproject.toml`:**
    ```toml
    [build-system]
    requires = ["maturin>=1.0,<2.0"]
    build-backend = "maturin"

    [project]
    name = "aqft_curved"
    version = "0.1.0"

    [tool.maturin]
    manifest-path = "rust/Cargo.toml"
    # NOTE: No 'features' key!
    ```

3.  **`aqft_curved/__init__.py`:**
    - Uses a relative import to find the Rust extension, which is now correctly packaged inside it.
    ```python
    try:
        from . import _core
        hello_from_rust = _core.hello_from_rust
    except ImportError:
        # ... fallback code
    ```
