# C++ Architecture

JoltGym's C++ core is organized into four modules: **Core Physics**, **MJCF Parser**, **Renderer**, and **Python Bindings**.

## Module Overview

```
src/
|-- core/               # Physics simulation
|   |-- joltgym_core.h/cpp        # Global init, shared job system
|   |-- physics_world.h/cpp       # PhysicsSystem wrapper
|   |-- articulation.h/cpp        # Articulated bodies with joints
|   |-- motor_controller.h/cpp    # Joint motor control
|   |-- state_extractor.h/cpp     # Observation extraction
|   |-- body_registry.h/cpp       # Name-to-ID mapping
|   +-- collision_layers.h/cpp    # Broadphase collision layers
|
|-- mjcf/               # MuJoCo XML parsing
|   |-- mjcf_parser.h/cpp         # XML parser (tinyxml2)
|   |-- mjcf_model.h/cpp          # Data structures
|   |-- mjcf_to_jolt.h/cpp        # MJCF -> Jolt conversion
|   +-- mjcf_defaults.h/cpp       # Default class system
|
|-- renderer/           # Vulkan rendering
|   |-- renderer.h/cpp            # Abstract renderer interface
|   |-- vulkan_context.h/cpp      # Vulkan device/instance
|   |-- swapchain_renderer.h/cpp  # Window rendering + ImGui
|   |-- offscreen_renderer.h/cpp  # Headless RGB output
|   |-- pipeline.h/cpp            # Graphics pipelines
|   |-- camera.h/cpp              # Camera control
|   |-- scene_sync.h/cpp          # Physics -> render sync
|   |-- mesh_primitives.h/cpp     # Mesh generation
|   +-- imgui_layer.h/cpp         # Debug UI
|
+-- bindings/           # pybind11 Python bindings
    |-- module.cpp                # Module definition
    |-- bind_env.cpp              # HalfCheetahCore
    |-- bind_humanoid.cpp         # HumanoidCore
    |-- bind_multi_agent.cpp      # MultiAgentEnv
    |-- bind_world.cpp            # PhysicsWorld, Articulation
    |-- bind_state.cpp            # StateExtractor
    |-- bind_renderer.cpp         # Renderer bindings
    +-- world_pool.h/cpp          # WorldPool (parallel envs)
```

## Data Flow

A typical simulation step flows through the following path:

```
Python env.step(action)
  |
  +-- pybind11 (zero-copy NumPy array)
       |
       +-- HalfCheetahCore::step(action)
            |
            |-- MotorController::SetAction()    # Decode normalized action
            |-- Articulation::ApplyActions()     # Apply torques to joints
            |-- PhysicsWorld::ApplyPassiveForces()  # Damping/stiffness
            |-- PhysicsWorld::Step(dt)           # Jolt physics step (x frame_skip)
            |-- StateExtractor::ExtractObs()     # Read qpos, qvel
            +-- Compute reward                   # forward_vel - ctrl_cost
```

## Build System

JoltGym uses CMake 3.20+ with `FetchContent` for all C++ dependencies:

```cmake
# Key targets
joltgym_core      # Core physics library (static)
joltgym_renderer  # Vulkan renderer (static, optional)
joltgym_native    # Python extension module (shared)
```

The Python package is built via `scikit-build-core`, which invokes CMake and packages the result as a pip-installable wheel.

## Key Design Decisions

### Single-Threaded Per-World

Each `PhysicsWorld` uses `JobSystemSingleThreaded` instead of Jolt's shared thread pool. This avoids contention when running multiple worlds in parallel (see [WorldPool](bindings.md#worldpool)).

### Zero-Copy Python Interface

All observation/action arrays pass through pybind11's NumPy buffer protocol without copying. The C++ side writes directly into pre-allocated NumPy arrays.

### MJCF-First

The engine loads MuJoCo XML files natively rather than defining its own model format. This provides compatibility with the large existing ecosystem of MuJoCo models.

### Deterministic Simulation

Given the same seed, JoltGym produces identical trajectories. State snapshots allow exact restore for deterministic resets.
