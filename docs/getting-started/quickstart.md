# Quick Start

This guide walks you through creating your first JoltGym environment, running a basic simulation, and training an RL agent.

## Basic Usage

### Create and Step an Environment

```python
import joltgym

# Create a HalfCheetah environment
env = joltgym.make("JoltGym/HalfCheetah-v0")

# Reset with a seed for reproducibility
obs, info = env.reset(seed=42)
print(f"Observation shape: {obs.shape}")  # (17,)
print(f"Action space: {env.action_space}")  # Box(-1.0, 1.0, (6,))

# Run a simulation loop
total_reward = 0
for step in range(1000):
    action = env.action_space.sample()  # random actions
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        obs, info = env.reset()
        total_reward = 0

print(f"Total reward: {total_reward:.3f}")
env.close()
```

### Available Environments

=== "HalfCheetah-v0"

    2D planar cheetah with 6 actuated joints.

    ```python
    env = joltgym.make("JoltGym/HalfCheetah-v0")
    ```

    | Property | Value |
    |---|---|
    | Observation | `Box(-inf, inf, (17,))` |
    | Action | `Box(-1, 1, (6,))` |
    | Reward | `forward_velocity - 0.1 * ctrl_cost` |

=== "Humanoid-v0"

    3D humanoid with 17 actuated joints and a free 6DOF root.

    ```python
    env = joltgym.make("JoltGym/Humanoid-v0")
    ```

    | Property | Value |
    |---|---|
    | Observation | `Box(-inf, inf, (45,))` |
    | Action | `Box(-0.4, 0.4, (17,))` |
    | Reward | `1.25 * forward_vel + 5.0 * healthy - 0.1 * ctrl_cost` |
    | Termination | root z outside [1.0, 2.0] |

=== "CheetahRace-v0"

    N multi-agent cheetahs in a shared physics world.

    ```python
    env = joltgym.make("JoltGym/CheetahRace-v0", num_agents=4)
    ```

    | Property | Value |
    |---|---|
    | Observation | `Box(-inf, inf, (N*17,))` |
    | Action | `Box(-1, 1, (N*6,))` |
    | Reward | sum of per-agent forward rewards |

### Reading Step Info

Each `step()` returns an `info` dict with additional metrics:

```python
obs, reward, terminated, truncated, info = env.step(action)

# HalfCheetah info keys:
print(info["x_position"])   # root body X position
print(info["x_velocity"])   # root body X velocity
print(info["reward_run"])   # forward reward component
print(info["reward_ctrl"])  # control cost component (negative)

# Humanoid adds:
print(info["z_position"])   # root body height
print(info["reward_forward"])

# CheetahRace adds:
print(info["per_agent_reward"])  # array of per-agent rewards
print(info["agent_0_x"])         # agent 0 X position
print(info["agent_0_xvel"])      # agent 0 X velocity
```

## Training with Stable-Baselines3

### Single Agent PPO

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import joltgym

num_envs = 8

def make_env(rank):
    def _init():
        env = joltgym.make("JoltGym/HalfCheetah-v0")
        env.reset(seed=42 + rank)
        return env
    return _init

train_env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(num_envs)]))

model = PPO("MlpPolicy", train_env, verbose=1,
            tensorboard_log="logs/halfcheetah_ppo")

model.learn(total_timesteps=500_000)
model.save("models/halfcheetah_ppo")
```

### Monitor Training

```bash
tensorboard --logdir logs/
```

## Vectorized Environments (C++ WorldPool)

For maximum throughput, use the C++ `WorldPool` which steps all environments in parallel native threads with the GIL released:

```python
import numpy as np
from joltgym.vector import JoltVectorEnv

# Create 64 parallel environments
envs = JoltVectorEnv(num_envs=64, model_path="python/joltgym/assets/half_cheetah.xml")

obs, info = envs.reset(seed=42)
print(f"Observation shape: {obs.shape}")  # (64, 17)

for step in range(1000):
    actions = np.random.uniform(-1, 1, (64, 6)).astype(np.float32)
    obs, rewards, dones, truncs, infos = envs.step(actions)
```

This achieves ~73K env-steps/sec on Apple Silicon -- see [Performance](../performance.md) for details.

## Next Steps

- [Environments](../python/environments.md) -- detailed environment specifications
- [Vectorized Environments](../python/vectorized.md) -- C++ WorldPool deep dive
- [Training Demos](../demos/training.md) -- full training scripts
- [C++ Architecture](../cpp/architecture.md) -- understand the internals
