# Vectorized Environments

JoltGym provides `JoltVectorEnv`, a high-performance vectorized environment that steps N independent physics worlds in parallel using native C++ threads.

Unlike Python-based vectorization (e.g., `SubprocVecEnv`), `JoltVectorEnv` wraps the C++ `WorldPool` class which:

- Maintains N independent `PhysicsSystem` instances
- Steps all environments in parallel via C++ `ParallelFor`
- Releases the GIL during the entire hot loop
- Returns batched NumPy arrays with zero-copy where possible

## Quick Usage

```python
import numpy as np
from joltgym.vector import JoltVectorEnv

envs = JoltVectorEnv(
    num_envs=64,
    model_path="python/joltgym/assets/half_cheetah.xml",
)

obs, infos = envs.reset(seed=42)

actions = np.random.uniform(-1, 1, (64, 6)).astype(np.float32)
obs, rewards, dones, truncs, infos = envs.step(actions)
```

## Threading Architecture

```
Python thread (GIL released)
  +-- C++ WorldPool::StepAll()
       +-- ParallelFor across min(hardware_concurrency, 16) OS threads
            |-- thread 0: step envs [0, chunk)
            |-- thread 1: step envs [chunk, 2*chunk)
            +-- thread N: step envs [last_chunk, num_envs)
```

## When to Use

| Scenario | Recommended Approach |
|---|---|
| Single environment | `joltgym.make(...)` |
| 2--8 parallel envs (SB3) | `SubprocVecEnv` with `joltgym.make` |
| 16+ parallel envs | `JoltVectorEnv` (C++ WorldPool) |
| Maximum throughput | `JoltVectorEnv` with 64--256 envs |

## API Reference

::: joltgym.vector.jolt_vector_env.JoltVectorEnv
    options:
      show_source: false
      members:
        - __init__
        - step
        - reset
        - close
