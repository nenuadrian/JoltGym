# Demos & Examples

JoltGym ships with several example scripts in the `examples/` directory covering basic usage, RL training, and video recording.

## Available Examples

| Script | Description |
|---|---|
| [`demo_halfcheetah.py`](#basic-demo) | Random-action HalfCheetah demo |
| [`demo_vectorized.py`](#vectorized-benchmark) | WorldPool throughput benchmark |
| [`train_ppo.py`](training.md#halfcheetah-training) | Train HalfCheetah with PPO |
| [`train_humanoid.py`](training.md#humanoid-training) | Train Humanoid with PPO |
| [`train_multiagent.py`](training.md#multi-agent-training) | Train CheetahRace with PPO |
| [`record_video.py`](recording.md#halfcheetah-recording) | Record HalfCheetah video |
| [`record_humanoid.py`](recording.md#humanoid-recording) | Record Humanoid video |
| [`record_race.py`](recording.md#cheetahrace-recording) | Record multi-agent race video |

## Basic Demo

`examples/demo_halfcheetah.py`

Runs 1000 steps of HalfCheetah with random actions, printing observations and rewards:

```bash
python examples/demo_halfcheetah.py
```

```python
import joltgym

env = joltgym.make("JoltGym/HalfCheetah-v0")
obs, info = env.reset(seed=42)

total_reward = 0
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if step % 100 == 0:
        print(f"  Step {step:4d} | reward={reward:7.3f} | "
              f"x_pos={info['x_position']:7.3f} | "
              f"x_vel={info['x_velocity']:7.3f}")

    if terminated or truncated:
        obs, info = env.reset()
        total_reward = 0

env.close()
```

## Vectorized Benchmark

`examples/demo_vectorized.py`

Benchmarks the C++ WorldPool with 64 parallel environments over 10,000 steps:

```bash
python examples/demo_vectorized.py
```

```python
import numpy as np
from joltgym.vector.jolt_vector_env import JoltVectorEnv

num_envs = 64
envs = JoltVectorEnv(num_envs, model_path="python/joltgym/assets/half_cheetah.xml")
obs, info = envs.reset(seed=42)

import time
start = time.time()
for step in range(10000):
    actions = np.random.uniform(-1, 1, (num_envs, 6)).astype(np.float32)
    obs, rewards, terms, truncs, infos = envs.step(actions)
elapsed = time.time() - start

print(f"Throughput: {num_envs * 10000 / elapsed:,.0f} env-steps/second")
```

## Prerequisites

All demos require JoltGym to be installed:

```bash
pip install -e .
```

Training demos additionally require:

```bash
pip install stable-baselines3 tensorboard
```

Recording demos additionally require:

```bash
pip install matplotlib
```
