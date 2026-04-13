# Python Bindings

The `joltgym_native` pybind11 module exposes the C++ core to Python. The bindings are organized across several source files in `src/bindings/`.

## Module Structure

```
src/bindings/
  |-- module.cpp           # pybind11 module definition
  |-- bind_env.cpp         # HalfCheetahCore
  |-- bind_humanoid.cpp    # HumanoidCore
  |-- bind_multi_agent.cpp # MultiAgentEnv
  |-- bind_world.cpp       # PhysicsWorld, Articulation
  |-- bind_state.cpp       # StateExtractor
  |-- bind_renderer.cpp    # Renderer (optional)
  +-- world_pool.h/cpp     # WorldPool (parallel envs)
```

## HalfCheetahCore

`src/bindings/bind_env.cpp`

Single HalfCheetah environment, fully implemented in C++.

```python
from joltgym import joltgym_native

core = joltgym_native.HalfCheetahCore(
    model_path="path/to/half_cheetah.xml",
    forward_reward_weight=1.0,
    ctrl_cost_weight=0.1,
)

obs = core.reset(seed=42, noise_scale=0.1)
obs, reward, terminated, truncated = core.step(action)

# Properties
core.get_obs_dim()       # 17
core.get_action_dim()    # 6
core.get_root_x()        # current X position
core.get_x_velocity()    # current X velocity
core.get_forward_reward() # last step's forward reward
core.get_ctrl_cost()     # last step's control cost
core.shutdown()
```

## HumanoidCore

`src/bindings/bind_humanoid.cpp`

Single Humanoid environment with health tracking.

```python
core = joltgym_native.HumanoidCore(
    model_path="path/to/humanoid.xml",
    forward_reward_weight=1.25,
    ctrl_cost_weight=0.1,
    healthy_reward=5.0,
    healthy_z_min=1.0,
    healthy_z_max=2.0,
)

obs = core.reset(seed=42, noise_scale=0.005)
obs, reward, terminated, truncated = core.step(action)

core.get_root_x()    # X position
core.get_root_z()    # Z position (height)
core.get_x_velocity()
```

The episode terminates when the root body's Z position falls outside `[healthy_z_min, healthy_z_max]`.

## MultiAgentEnv

`src/bindings/bind_multi_agent.cpp`

N independent cheetahs in a single shared `PhysicsWorld`.

```python
env = joltgym_native.MultiAgentEnv(
    num_agents=4,
    model_path="path/to/half_cheetah.xml",
    agent_spacing=3.0,
    forward_reward_weight=1.0,
    ctrl_cost_weight=0.1,
)

obs = env.reset_all(seed=42, noise_scale=0.1)
obs, rewards = env.step(actions)  # actions: (N, 6), obs: (N, 17), rewards: (N,)

env.obs_dim   # 17 (per agent)
env.act_dim   # 6 (per agent)
env.get_agent_x(i)           # X position of agent i
env.get_agent_x_velocity(i)  # X velocity of agent i
```

## WorldPool

`src/bindings/world_pool.h`

N parallel environments stepped in C++ threads. This is the performance-critical class for high-throughput training.

```python
pool = joltgym_native.WorldPool(
    num_envs=64,
    model_path="path/to/half_cheetah.xml",
    forward_reward_weight=1.0,
    ctrl_cost_weight=0.1,
)

obs = pool.reset_all(seed=42)
obs, rewards, dones = pool.step_all(actions)

pool.obs_dim     # 17
pool.act_dim     # 6
pool.num_envs    # 64
```

### C++ Implementation

```cpp
class WorldPool {
public:
    WorldPool(int num_envs, const std::string& model_path,
              float forward_reward_weight = 1.0f,
              float ctrl_cost_weight = 0.1f);

    // Parallel step: distributes work across OS threads
    void StepAll(const float* actions, float* obs_out,
                 float* rewards_out, bool* dones_out);

    // Parallel reset
    void ResetAll(float* obs_out,
                  std::optional<uint32_t> base_seed = std::nullopt,
                  float noise_scale = 0.1f);

    // Single-env reset
    void ResetOne(int idx, float* obs_out,
                  std::optional<uint32_t> seed = std::nullopt,
                  float noise_scale = 0.1f);

    int num_envs() const;
    int obs_dim() const;
    int act_dim() const;
};
```

### Per-Environment State

Each environment in the pool maintains independent state:

```cpp
struct EnvInstance {
    PhysicsWorld world;
    Articulation* articulation;
    std::unique_ptr<StateExtractor> state;
    std::vector<float> obs_buffer;
    std::vector<float> action_buffer;

    float dt = 0.01f;
    int frame_skip = 5;
    float forward_reward_weight = 1.0f;
    float ctrl_cost_weight = 0.1f;

    // Per-step outputs
    float x_velocity, forward_reward, ctrl_cost, reward;
    int episode_length;
    std::mt19937 rng;
};
```

### Threading Model

The pool uses `min(hardware_concurrency, 16)` OS threads. The GIL is released during `StepAll` and `ResetAll`, so the Python interpreter is not blocked during physics computation.

```
StepAll(actions) with GIL released:
  thread 0: for i in [0, chunk):     StepEnv(i, actions)
  thread 1: for i in [chunk, 2*chunk): StepEnv(i, actions)
  ...
  thread N: for i in [last, num_envs): StepEnv(i, actions)
```

Each call to `StepEnv` performs:

1. Copy actions to the environment's action buffer
2. Apply actions to motor controllers
3. Apply passive forces (damping, stiffness)
4. Step physics `frame_skip` times
5. Extract observations
6. Compute reward
7. Check termination and auto-reset if needed

## Low-Level Bindings

`src/bindings/bind_world.cpp` and `src/bindings/bind_state.cpp` expose lower-level access:

```python
# PhysicsWorld
world = joltgym_native.PhysicsWorld()
world.init(max_bodies=2048, single_threaded=True)
world.step(dt=0.01, collision_steps=1)
world.save_snapshot()
world.restore_snapshot()

# Articulation
art = world.get_articulation(0)
art.get_qpos_dim()
art.get_qvel_dim()
art.get_root_x(world.body_interface)

# StateExtractor
state = joltgym_native.StateExtractor(art, world, qpos_skip=1)
obs = state.extract_obs()
```

These are primarily useful for debugging and custom environment development.
