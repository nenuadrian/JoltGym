# Core Physics

The core physics module wraps Jolt Physics and provides articulated body simulation, motor control, and state extraction.

All classes live in the `joltgym` namespace.

## PhysicsWorld

Central simulation container wrapping Jolt's `PhysicsSystem` with snapshot support and articulation management.

::: doxy.joltgym-cpp.Class.joltgym::PhysicsWorld

---

## StateSnapshot

Complete snapshot of all body states for deterministic reset.

::: doxy.joltgym-cpp.Class.joltgym::StateSnapshot

---

## Articulation

Articulated body — a tree of rigid bodies connected by motorized joints.

::: doxy.joltgym-cpp.Class.joltgym::Articulation

---

## RootDOF

Describes a single root degree of freedom (planar or free).

::: doxy.joltgym-cpp.Class.joltgym::RootDOF

---

## MotorController

Single-joint motor controller mapping normalized actions to torques via Jolt's implicit spring integrator.

::: doxy.joltgym-cpp.Class.joltgym::MotorController

---

## StateExtractor

Extracts observation vectors (`qpos`, `qvel`) from physics state.

::: doxy.joltgym-cpp.Class.joltgym::StateExtractor
