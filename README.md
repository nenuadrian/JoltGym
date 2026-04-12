# JoltGym

A MuJoCo-compatible physics simulation engine for reinforcement learning, built on [Jolt Physics](https://github.com/jrouwe/JoltPhysics) with Vulkan rendering and Python (Gymnasium) bindings.

## Features

- **Jolt Physics backend** — C++17, deterministic (cross-platform), excellent multicore scaling
- **MJCF compatibility** — loads MuJoCo XML models (HalfCheetah included)
- **Gymnasium API** — drop-in `env.reset()` / `env.step()` / `env.render()` interface
- **Vulkan renderer** — SDL2 window + Dear ImGui debug overlay (macOS via MoltenVK, Linux, Windows)
- **Offscreen rendering** — headless `rgb_array` mode for training servers
- **Multi-instance** — N parallel `PhysicsSystem` instances with shared thread pool
- **Multi-agent** — multiple robots in a single world with collision control
- **Zero-copy Python** — pybind11 + NumPy, hot path entirely in C++

## Architecture

```
Python (Gymnasium API)
  └── pybind11 bindings (zero-copy NumPy)
       └── C++ Core
            ├── MJCF Parser (tinyxml2)
            ├── Physics (Jolt Physics)
            ├── Renderer (Vulkan + SDL2 + ImGui)
            └── WorldPool (N parallel PhysicsSystem instances)
```

## Requirements

- CMake 3.20+
- C++17 compiler (Clang, GCC, MSVC)
- Python 3.9+ with NumPy and Gymnasium
- Vulkan SDK (for renderer; optional)

All C++ dependencies are fetched automatically via CMake FetchContent:
Jolt Physics, tinyxml2, SDL2, Dear ImGui, vk-bootstrap, VMA.

## Build

### Python package (recommended)

```bash
pip install -e .
```

This uses scikit-build-core to build the C++ code and install the Python package.

### CMake (C++ development)

```bash
# Full build (with renderer)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Without Vulkan renderer
cmake -B build -DCMAKE_BUILD_TYPE=Release -DJOLTGYM_BUILD_RENDERER=OFF
cmake --build build -j$(nproc)

# With C++ tests
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DJOLTGYM_BUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/tests/cpp/test_basic
```

### CMake options

| Option | Default | Description |
|---|---|---|
| `JOLTGYM_BUILD_RENDERER` | `ON` | Build Vulkan renderer (requires Vulkan SDK) |
| `JOLTGYM_BUILD_PYTHON` | `ON` | Build pybind11 Python module |
| `JOLTGYM_BUILD_TESTS` | `OFF` | Build C++ test executables |

## Usage

```python
import joltgym

env = joltgym.make("JoltGym/HalfCheetah-v0")
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Environment details (HalfCheetah-v0)

| Property | Value |
|---|---|
| Observation | `Box(-inf, inf, (17,))` — qpos[1:] + qvel |
| Action | `Box(-1, 1, (6,))` — normalized joint torques |
| Reward | `forward_velocity - 0.1 * ctrl_cost` |
| Timestep | 0.01s, frame_skip=5 |

### Running from CMake build (without pip install)

```bash
PYTHONPATH=build/src/bindings:python python3 examples/demo_halfcheetah.py
```

## Tests

```bash
# Python
PYTHONPATH=build/src/bindings:python python3 -m pytest tests/python/ -v

# C++ (requires -DJOLTGYM_BUILD_TESTS=ON)
./build/tests/cpp/test_basic
```

## License

MIT
