# MJCF Parser

The MJCF module parses MuJoCo XML model files and converts them into Jolt Physics bodies and constraints.

```
half_cheetah.xml
      |
      v
  MjcfParser::Parse()     -- XML -> MjcfModel
      |
      v
  MjcfToJolt::Build()     -- MjcfModel -> Jolt bodies/constraints
      |
      v
  PhysicsWorld + Articulation
```

## Multi-Joint Decomposition

Bodies with multiple hinge joints (e.g., humanoid hip with 3 DOFs) are decomposed into a chain of intermediate bodies:

```
parent -> hinge_1 -> intermediate_1 -> hinge_2 -> intermediate_2 -> hinge_3 -> actual_body
```

Intermediate bodies are tiny-mass spheres (0.001 kg) that act purely as kinematic connectors.

## Supported MJCF Features

| Feature | Status |
|---|---|
| Body hierarchy | Supported |
| Hinge joints | Supported |
| Slide joints | Supported |
| Free joints (6DOF) | Supported |
| Ball joints | Parsed, converted to 3 hinges |
| Capsule/Sphere/Box/Cylinder/Plane geoms | Supported |
| Actuators (motor) | Supported |
| Default classes | Supported |
| Compiler directives | Supported |
| `fromto` syntax | Supported |
| Tendons | Parsed (not physically simulated) |
| Contacts/exclude | Not yet supported |

## Data Model

### MjcfModel

::: doxy.joltgym-cpp.Class.joltgym::MjcfModel

### MjcfBody

::: doxy.joltgym-cpp.Class.joltgym::MjcfBody

### MjcfJoint

::: doxy.joltgym-cpp.Class.joltgym::MjcfJoint

### MjcfGeom

::: doxy.joltgym-cpp.Class.joltgym::MjcfGeom

### MjcfActuator

::: doxy.joltgym-cpp.Class.joltgym::MjcfActuator

### MjcfCompiler

::: doxy.joltgym-cpp.Class.joltgym::MjcfCompiler

### MjcfOption

::: doxy.joltgym-cpp.Class.joltgym::MjcfOption
