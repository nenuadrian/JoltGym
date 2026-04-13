# Python API Reference

## Module: `joltgym`

### `joltgym.make`

```python
def make(env_id: str, **kwargs) -> gym.Env
```

Create a JoltGym environment by ID. Wraps `gymnasium.make()` with JoltGym's registered environments.

**Registered environment IDs:**

- `"JoltGym/HalfCheetah-v0"`
- `"JoltGym/Humanoid-v0"`
- `"JoltGym/CheetahRace-v0"`

```python
import joltgym

env = joltgym.make("JoltGym/HalfCheetah-v0")
env = joltgym.make("JoltGym/Humanoid-v0", healthy_z_min=0.8)
env = joltgym.make("JoltGym/CheetahRace-v0", num_agents=4)
```

---

## Class: `HalfCheetahEnv`

`joltgym.envs.HalfCheetahEnv`

A Gymnasium environment for 2D planar cheetah locomotion.

### Constructor

```python
HalfCheetahEnv(
    render_mode: str = None,
    forward_reward_weight: float = 1.0,
    ctrl_cost_weight: float = 0.1,
    reset_noise_scale: float = 0.1,
)
```

### Methods

#### `step(action) -> tuple`

Take one step in the environment.

- **action** (`np.ndarray`, shape `(6,)`) -- Normalized joint torques in `[-1, 1]`.
- **Returns:** `(obs, reward, terminated, truncated, info)`
    - `obs` -- `np.ndarray` of shape `(17,)`
    - `reward` -- `float`
    - `terminated` -- `bool` (always `False` for HalfCheetah)
    - `truncated` -- `bool` (always `False`)
    - `info` -- `dict` with keys: `x_position`, `x_velocity`, `reward_run`, `reward_ctrl`

#### `reset(*, seed=None, options=None) -> tuple`

Reset the environment to initial state with optional noise.

- **seed** (`int`, optional) -- Random seed for reproducibility.
- **Returns:** `(obs, info)`

#### `close()`

Shut down the underlying C++ physics engine.

---

## Class: `HumanoidEnv`

`joltgym.envs.HumanoidEnv`

A Gymnasium environment for 3D bipedal humanoid locomotion.

### Constructor

```python
HumanoidEnv(
    render_mode: str = None,
    forward_reward_weight: float = 1.25,
    ctrl_cost_weight: float = 0.1,
    healthy_reward: float = 5.0,
    healthy_z_min: float = 1.0,
    healthy_z_max: float = 2.0,
    reset_noise_scale: float = 0.005,
)
```

### Methods

#### `step(action) -> tuple`

- **action** (`np.ndarray`, shape `(17,)`) -- Normalized joint torques in `[-0.4, 0.4]`.
- **Returns:** `(obs, reward, terminated, truncated, info)`
    - `obs` -- `np.ndarray` of shape `(45,)`
    - `terminated` -- `True` when root Z is outside `[healthy_z_min, healthy_z_max]`
    - `info` -- `dict` with keys: `x_position`, `z_position`, `x_velocity`, `reward_forward`, `reward_ctrl`

#### `reset(*, seed=None, options=None) -> tuple`

Same interface as `HalfCheetahEnv.reset()`.

#### `close()`

Shut down the underlying C++ physics engine.

---

## Class: `CheetahRaceEnv`

`joltgym.envs.CheetahRaceEnv`

A multi-agent environment with N cheetahs in a shared physics world.

### Constructor

```python
CheetahRaceEnv(
    num_agents: int = 2,
    render_mode: str = None,
    agent_spacing: float = 3.0,
    forward_reward_weight: float = 1.0,
    ctrl_cost_weight: float = 0.1,
    reset_noise_scale: float = 0.1,
)
```

### Methods

#### `step(action) -> tuple`

- **action** (`np.ndarray`, shape `(num_agents * 6,)`) -- Flat concatenation of per-agent actions.
- **Returns:** `(obs, reward, terminated, truncated, info)`
    - `obs` -- shape `(num_agents * 17,)`
    - `reward` -- `float` (sum of all agents' rewards)
    - `info["per_agent_reward"]` -- `np.ndarray` of shape `(num_agents,)`

#### `reset(*, seed=None, options=None) -> tuple`

Reset all agents to initial positions.

#### `get_per_agent_obs(flat_obs) -> np.ndarray`

Split flat observation into per-agent arrays. Returns shape `(num_agents, 17)`.

#### `get_per_agent_actions(flat_action) -> np.ndarray`

Split flat action into per-agent arrays. Returns shape `(num_agents, 6)`.

---

## Class: `JoltVectorEnv`

`joltgym.vector.JoltVectorEnv`

High-performance vectorized environment using C++ parallel stepping.

See [Vectorized Environments](vectorized.md) for full documentation.

### Constructor

```python
JoltVectorEnv(
    num_envs: int,
    model_path: str,
    forward_reward_weight: float = 1.0,
    ctrl_cost_weight: float = 0.1,
)
```

### Methods

#### `step(actions) -> tuple`

- **actions** (`np.ndarray`, shape `(num_envs, act_dim)`) -- Batched actions.
- **Returns:** `(obs, rewards, dones, truncs, infos)`

#### `reset(*, seed=None, options=None) -> tuple`

- **Returns:** `(obs, infos)`

---

## Native Module: `joltgym_native`

The `joltgym_native` module is the pybind11 C++ extension. It is not intended for direct use -- prefer the Python wrapper classes above. Key classes exposed:

| Class | Description |
|---|---|
| `HalfCheetahCore` | Single HalfCheetah environment (C++) |
| `HumanoidCore` | Single Humanoid environment (C++) |
| `MultiAgentEnv` | Multi-agent shared-world environment (C++) |
| `WorldPool` | N parallel environments with C++ threading |
| `PhysicsWorld` | Low-level physics world access |
| `Articulation` | Articulated body with joints and motors |
