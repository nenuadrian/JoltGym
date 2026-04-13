# Vectorized Environments

JoltGym provides `JoltVectorEnv`, a high-performance vectorized environment that steps N independent physics worlds in parallel using native C++ threads.

## Overview

Unlike Python-based vectorization (e.g., `SubprocVecEnv`), `JoltVectorEnv` wraps the C++ `WorldPool` class which:

- Maintains N independent `PhysicsSystem` instances
- Steps all environments in parallel via C++ `ParallelFor`
- Releases the GIL during the entire hot loop (action apply, physics step, observation extraction, reward computation)
- Returns batched NumPy arrays with zero-copy where possible

This eliminates Python subprocess overhead and achieves ~73K env-steps/sec at 256 environments on Apple Silicon.

## Usage

```python
import numpy as np
from joltgym.vector import JoltVectorEnv

# Create 64 parallel HalfCheetah environments
envs = JoltVectorEnv(
    num_envs=64,
    model_path="python/joltgym/assets/half_cheetah.xml",
)

# Reset all environments
obs, infos = envs.reset(seed=42)
print(obs.shape)  # (64, 17)

# Step all environments
actions = np.random.uniform(-1, 1, (64, 6)).astype(np.float32)
obs, rewards, dones, truncs, infos = envs.step(actions)
print(rewards.shape)  # (64,)
print(dones.shape)    # (64,)
```

## API Reference

### `JoltVectorEnv.__init__`

```python
JoltVectorEnv(
    num_envs: int,
    model_path: str,
    forward_reward_weight: float = 1.0,
    ctrl_cost_weight: float = 0.1,
)
```

| Parameter | Description |
|---|---|
| `num_envs` | Number of parallel environments |
| `model_path` | Path to the MJCF XML model file |
| `forward_reward_weight` | Weight on forward velocity reward |
| `ctrl_cost_weight` | Weight on control cost penalty |

### `JoltVectorEnv.step`

```python
def step(self, actions: np.ndarray) -> tuple:
    """Step all environments.

    Args:
        actions: Array of shape (num_envs, act_dim), dtype float32.

    Returns:
        obs: (num_envs, obs_dim) float32
        rewards: (num_envs,) float32
        dones: (num_envs,) bool -- True if episode ended (auto-reset applied)
        truncs: (num_envs,) bool -- always False (no truncation)
        infos: list of empty dicts
    """
```

### `JoltVectorEnv.reset`

```python
def reset(self, *, seed=None, options=None) -> tuple:
    """Reset all environments.

    Args:
        seed: Optional base seed. Environment i gets seed + i.

    Returns:
        obs: (num_envs, obs_dim) float32
        infos: list of empty dicts
    """
```

### Properties

| Property | Type | Description |
|---|---|---|
| `num_envs` | `int` | Number of parallel environments |
| `observation_space` | `Box` | Batched observation space `(num_envs, obs_dim)` |
| `action_space` | `Box` | Batched action space `(num_envs, act_dim)` |
| `single_observation_space` | `Box` | Single-environment observation space `(obs_dim,)` |
| `single_action_space` | `Box` | Single-environment action space `(act_dim,)` |

## Auto-Reset

When an environment in the pool reaches a terminal state, it is automatically reset and the next observation is from the new episode. The `dones` flag indicates which environments were reset.

## Benchmarking

```python
import time
import numpy as np
from joltgym.vector import JoltVectorEnv

num_envs = 256
num_steps = 10000

envs = JoltVectorEnv(num_envs=num_envs,
                     model_path="python/joltgym/assets/half_cheetah.xml")
obs, _ = envs.reset(seed=42)

start = time.time()
for _ in range(num_steps):
    actions = np.random.uniform(-1, 1, (num_envs, 6)).astype(np.float32)
    obs, rewards, dones, truncs, infos = envs.step(actions)
elapsed = time.time() - start

total_steps = num_envs * num_steps
print(f"Throughput: {total_steps / elapsed:,.0f} env-steps/sec")
```

## Threading Architecture

```
Python thread (GIL released)
  +-- C++ WorldPool::StepAll()
       +-- ParallelFor across min(hardware_concurrency, 16) OS threads
            |-- thread 0: step envs [0, chunk)
            |-- thread 1: step envs [chunk, 2*chunk)
            |-- ...
            +-- thread N: step envs [last_chunk, num_envs)
```

Each environment uses `JobSystemSingleThreaded` (no internal Jolt threading), avoiding contention between worlds. This design scales linearly up to the number of hardware threads.

## When to Use

| Scenario | Recommended Approach |
|---|---|
| Single environment | `joltgym.make(...)` |
| 2-8 parallel envs (SB3) | `SubprocVecEnv` with `joltgym.make` |
| 16+ parallel envs | `JoltVectorEnv` (C++ WorldPool) |
| Maximum throughput | `JoltVectorEnv` with 64-256 envs |
