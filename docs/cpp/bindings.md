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

::: doxy.joltgym-cpp.Class.joltgym::WorldPool

---

## EnvInstance

Per-environment state within the WorldPool.

::: doxy.joltgym-cpp.Class.joltgym::EnvInstance

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
