# Installation

## Requirements

- **CMake** 3.20+
- **C++17 compiler** (Clang, GCC, or MSVC)
- **Python** 3.9+ with NumPy and Gymnasium
- **Vulkan SDK** (optional, for the renderer)

All C++ dependencies are fetched automatically via CMake `FetchContent`:

| Dependency | Purpose |
|---|---|
| [Jolt Physics](https://github.com/jrouwe/JoltPhysics) | Physics engine |
| [tinyxml2](https://github.com/leethomason/tinyxml2) | MJCF XML parsing |
| [SDL2](https://www.libsdl.org/) | Windowing (renderer) |
| [Dear ImGui](https://github.com/ocornut/imgui) | Debug UI overlay |
| [vk-bootstrap](https://github.com/charles-lunarg/vk-bootstrap) | Vulkan initialization |
| [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | Vulkan memory allocation |

## Python Package (Recommended)

The simplest way to install JoltGym is as a Python package using `pip`:

```bash
pip install -e .
```

This uses [scikit-build-core](https://github.com/scikit-build/scikit-build-core) to build the C++ extension module and install the Python package in one step.

### Optional Dependencies

For development and testing:

```bash
pip install -e ".[dev]"
```

This installs additional packages:

- `pytest` -- for running the test suite
- `mujoco` -- for comparison benchmarks

For training with PPO (used in demos):

```bash
pip install stable-baselines3 tensorboard
```

For video recording:

```bash
pip install matplotlib
# Optional: install ffmpeg for MP4 output (otherwise GIF)
brew install ffmpeg    # macOS
sudo apt install ffmpeg  # Ubuntu/Debian
```

## CMake Build (C++ Development)

For working on the C++ source directly:

### Full Build (with renderer)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Without Vulkan Renderer

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DJOLTGYM_BUILD_RENDERER=OFF
cmake --build build -j$(nproc)
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `JOLTGYM_BUILD_RENDERER` | `ON` | Build Vulkan renderer (requires Vulkan SDK) |
| `JOLTGYM_BUILD_PYTHON` | `ON` | Build pybind11 Python module |
| `JOLTGYM_BUILD_TESTS` | `OFF` | Build C++ test executables |

## Verifying the Installation

```python
import joltgym

env = joltgym.make("JoltGym/HalfCheetah-v0")
obs, info = env.reset(seed=42)
print(f"Observation shape: {obs.shape}")  # (17,)
print(f"Action space: {env.action_space}")  # Box(-1.0, 1.0, (6,))
env.close()
print("JoltGym is working!")
```

## Platform Notes

### macOS

Vulkan rendering uses MoltenVK. Install the [Vulkan SDK for macOS](https://vulkan.lunarg.com/sdk/home#mac) if you want the renderer.

### Linux

Install the Vulkan SDK and development headers:

```bash
sudo apt install libvulkan-dev vulkan-tools
```

### Windows

Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) and ensure Visual Studio 2019+ is available with C++17 support.
