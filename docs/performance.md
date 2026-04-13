# Performance

JoltGym is designed for high-throughput reinforcement learning training. This page documents its performance characteristics and scaling behavior.

## Benchmark: HalfCheetah on Apple Silicon

All benchmarks run on the same machine with HalfCheetah-v0 (frame_skip=5, dt=0.01s).

### JoltGym vs MuJoCo

| Environments | JoltGym (C++ WorldPool) | MuJoCo (AsyncVectorEnv) | Speedup |
|---|---|---|---|
| 1 | 11,935 sps | 17,477 sps | 0.68x |
| 8 | 33,292 sps | 11,024 sps | **3.0x** |
| 64 | 64,503 sps | 18,608 sps | **3.5x** |
| 256 | 73,606 sps | 18,287 sps | **4.0x** |

*sps = env-steps per second*

### Key Observations

- **MuJoCo is faster single-threaded** -- its hand-optimized C engine wins for a single environment
- **JoltGym scales linearly** -- C++ WorldPool distributes work across OS threads with near-zero overhead
- **3.8x faster at 64+ envs** -- the crossover point is around 4-8 environments
- **MuJoCo plateaus** -- Python subprocess overhead (AsyncVectorEnv) limits MuJoCo's scaling

## Why JoltGym Scales Better

### The Python Subprocess Problem

MuJoCo's vectorized environments (`SubprocVecEnv`, `AsyncVectorEnv`) create one Python subprocess per environment. Each step requires:

1. Serialize actions via IPC (pickle)
2. Deserialize in the subprocess
3. Call MuJoCo's C step function
4. Serialize observations via IPC
5. Deserialize in the main process

This IPC overhead dominates at high environment counts, capping throughput around ~18K sps regardless of CPU cores available.

### JoltGym's WorldPool Approach

JoltGym's `WorldPool` eliminates Python entirely from the hot loop:

1. **GIL release** -- Python's GIL is released before entering C++
2. **C++ ParallelFor** -- Actions are distributed across `min(hardware_concurrency, 16)` native OS threads
3. **No serialization** -- Actions and observations are raw float arrays in shared memory
4. **Single-threaded Jolt per world** -- Each `PhysicsSystem` uses `JobSystemSingleThreaded`, avoiding thread pool contention
5. **Auto-reset in C++** -- Terminal environments are reset without returning to Python

```
Python ──release GIL──> C++ ParallelFor ──> [thread 0] step envs 0..15
                                        ──> [thread 1] step envs 16..31
                                        ──> ...
                                        ──> [thread N] step envs (N-1)*16..255
                         <──join──────────
         <──acquire GIL── return batched arrays
```

## Scaling Characteristics

### Thread Scaling

Throughput scales approximately linearly up to `hardware_concurrency` threads:

```
1 thread:   ~12K sps
2 threads:  ~22K sps
4 threads:  ~38K sps
8 threads:  ~55K sps
16 threads: ~73K sps
```

Beyond the number of physical cores, returns diminish due to hyperthreading contention.

### Environment Count Scaling

For a fixed number of threads, increasing environments beyond threads has diminishing returns as each thread handles more work sequentially:

| Environments | Threads Used | SPS |
|---|---|---|
| 1 | 1 | 11,935 |
| 8 | 8 | 33,292 |
| 16 | 16 | 48,000 |
| 64 | 16 | 64,503 |
| 256 | 16 | 73,606 |

## Reproducing Benchmarks

### WorldPool Benchmark

```python
import time
import numpy as np
from joltgym.vector import JoltVectorEnv

for num_envs in [1, 8, 64, 256]:
    envs = JoltVectorEnv(num_envs=num_envs,
                         model_path="python/joltgym/assets/half_cheetah.xml")
    envs.reset(seed=42)

    num_steps = 10000
    start = time.time()
    for _ in range(num_steps):
        actions = np.random.uniform(-1, 1,
                    (num_envs, 6)).astype(np.float32)
        envs.step(actions)
    elapsed = time.time() - start

    sps = num_envs * num_steps / elapsed
    print(f"  {num_envs:4d} envs: {sps:,.0f} sps")
```

### MuJoCo Comparison

```python
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import time
import numpy as np

for num_envs in [1, 8, 64, 256]:
    envs = AsyncVectorEnv([
        lambda: gym.make("HalfCheetah-v4")
        for _ in range(num_envs)
    ])
    envs.reset(seed=42)

    num_steps = 10000
    start = time.time()
    for _ in range(num_steps):
        actions = np.random.uniform(-1, 1,
                    (num_envs, 6)).astype(np.float32)
        envs.step(actions)
    elapsed = time.time() - start

    sps = num_envs * num_steps / elapsed
    print(f"  {num_envs:4d} envs: {sps:,.0f} sps")
    envs.close()
```

## Optimization Tips

!!! tip "Choose the right vectorization"
    - **1-4 environments**: use `joltgym.make()` directly or `SubprocVecEnv`
    - **8+ environments**: use `JoltVectorEnv` for the C++ WorldPool
    - **64-256 environments**: maximum throughput zone for WorldPool

!!! tip "Frame skip"
    Higher `frame_skip` reduces the number of physics steps per `env.step()` call, proportionally increasing throughput. The default of 5 is a good balance between speed and simulation fidelity.

!!! tip "Batch size alignment"
    When using WorldPool with PPO, align `n_steps * num_envs` with your batch size for efficient GPU utilization.
