# JoltGym

A high-performance, MuJoCo-compatible physics simulation engine for reinforcement learning, built on [Jolt Physics](https://github.com/jrouwe/JoltPhysics) with Vulkan rendering and Python (Gymnasium) bindings.

![HalfCheetah-v0](assets/cheetah.png)

## Key Features

- **Jolt Physics backend** -- C++17, deterministic, excellent multicore scaling
- **MJCF compatibility** -- loads MuJoCo XML models (HalfCheetah, Humanoid)
- **Gymnasium API** -- drop-in `env.reset()` / `env.step()` / `env.render()` interface
- **Vulkan renderer** -- SDL2 window + Dear ImGui debug overlay (macOS via MoltenVK, Linux, Windows)
- **Offscreen rendering** -- headless `rgb_array` mode for training servers
- **WorldPool** -- N parallel `PhysicsSystem` instances stepped from C++, ~73K env-steps/sec
- **Multi-agent** -- multiple robots in a single shared world with collision
- **Zero-copy Python** -- pybind11 + NumPy, hot path entirely in C++

## Architecture Overview

```
Python (Gymnasium API)
  +-- pybind11 bindings (zero-copy NumPy)
       +-- C++ Core
            |-- MJCF Parser (tinyxml2)
            |-- Physics (Jolt Physics)
            |-- Renderer (Vulkan + SDL2 + ImGui)
            +-- WorldPool (N parallel PhysicsSystem instances)
```

## Environments

| Environment | Description | Obs Dim | Act Dim |
|---|---|---|---|
| `JoltGym/HalfCheetah-v0` | 2D planar cheetah locomotion | 17 | 6 |
| `JoltGym/Humanoid-v0` | 3D bipedal humanoid locomotion | 45 | 17 |
| `JoltGym/CheetahRace-v0` | N multi-agent cheetahs racing | N*17 | N*6 |

## Quick Example

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

## Performance

JoltGym scales to **3.8x faster than MuJoCo at 64+ parallel environments** via C++ parallel stepping, eliminating Python subprocess overhead entirely.

| Environments | JoltGym (C++ WorldPool) | MuJoCo (AsyncVectorEnv) |
|---|---|---|
| 1 | 11,935 sps | 17,477 sps |
| 8 | 33,292 sps | 11,024 sps |
| 64 | 64,503 sps | 18,608 sps |
| 256 | 73,606 sps | 18,287 sps |

See [Performance](performance.md) for full benchmarks and analysis.

## License

MIT
