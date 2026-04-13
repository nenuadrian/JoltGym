# Python Bindings

The `joltgym_native` pybind11 module exposes the C++ core to Python. The bindings are organized across several source files in `src/bindings/`.

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

## WorldPool

The performance-critical class for high-throughput training. Maintains N independent physics worlds stepped in parallel via native OS threads.

The MJCF model is parsed once and used to build N independent PhysicsWorld instances, each with its own single-threaded Jolt JobSystem. `StepAll()` distributes environments across `min(hardware_concurrency, 16)` OS threads via ParallelFor, with the Python GIL released for the entire duration.

Achieves ~73K env-steps/sec at 256 environments on Apple Silicon.

```cpp
class WorldPool {
public:
    WorldPool(int num_envs, const std::string& model_path,
              float forward_reward_weight = 1.0f,
              float ctrl_cost_weight = 0.1f);

    int num_envs() const;
    int obs_dim() const;
    int act_dim() const;

    /// Step all environments in parallel (GIL released).
    /// actions:     [num_envs * act_dim]
    /// obs_out:     [num_envs * obs_dim]
    /// rewards_out: [num_envs]
    /// dones_out:   [num_envs] — true indicates auto-reset
    void StepAll(const float* actions, float* obs_out,
                 float* rewards_out, bool* dones_out);

    /// Reset all environments. Env i gets seed base_seed + i.
    void ResetAll(float* obs_out, std::optional<uint32_t> base_seed = std::nullopt,
                  float noise_scale = 0.1f);

    /// Reset a single environment by index.
    void ResetOne(int idx, float* obs_out,
                  std::optional<uint32_t> seed = std::nullopt,
                  float noise_scale = 0.1f);
};
```

---

## EnvInstance

Per-environment state within the WorldPool. Each EnvInstance owns its own PhysicsWorld (with single-threaded Jolt), allowing contention-free parallel stepping.

```cpp
struct EnvInstance {
    PhysicsWorld world;                     // Per-env physics system
    Articulation* articulation = nullptr;   // The robot articulation
    std::unique_ptr<StateExtractor> state;  // Observation extractor
    std::vector<float> obs_buffer;          // Pre-allocated observation buffer
    std::vector<float> action_buffer;       // Pre-allocated action buffer

    float dt = 0.01f;                       // Physics timestep (seconds)
    int frame_skip = 5;                     // Physics steps per env step
    float forward_reward_weight = 1.0f;     // Forward velocity reward weight
    float ctrl_cost_weight = 0.1f;          // Control cost penalty weight

    int obs_dim = 0;
    int act_dim = 0;
    int episode_length = 0;
    float x_velocity = 0;
    float forward_reward = 0;
    float ctrl_cost = 0;
    float reward = 0;

    std::mt19937 rng{42};                   // Per-env RNG for resets
};
```

---

## Threading Model

The pool uses `min(hardware_concurrency, 16)` OS threads. The GIL is released during `StepAll` and `ResetAll`.

```
StepAll(actions) with GIL released:
  thread 0: for i in [0, chunk):       StepEnv(i, actions)
  thread 1: for i in [chunk, 2*chunk): StepEnv(i, actions)
  ...
  thread N: for i in [last, num_envs): StepEnv(i, actions)
```

## Native Classes Summary

The following classes are exposed to Python via pybind11 but are not intended for direct use — prefer the Python wrapper classes in `joltgym.envs` and `joltgym.vector`.

| Class | Description |
|---|---|
| `HalfCheetahCore` | Single HalfCheetah environment (C++) |
| `HumanoidCore` | Single Humanoid environment (C++) |
| `MultiAgentEnv` | Multi-agent shared-world environment (C++) |
| `WorldPool` | N parallel environments with C++ threading |
| `PhysicsWorld` | Low-level physics world access |
| `Articulation` | Articulated body with joints and motors |
