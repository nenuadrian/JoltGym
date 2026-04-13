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

Top-level parsed representation of a MuJoCo XML model.

```cpp
struct MjcfModel {
    std::string name;                     // Model name
    MjcfCompiler compiler;                // Compiler directives
    MjcfOption option;                    // Simulation options
    MjcfBody worldbody;                   // Root of the body tree
    std::vector<MjcfActuator> actuators;  // Motor definitions
};
```

### MjcfBody

A body element parsed from `<body>`, forming a tree hierarchy.

```cpp
struct MjcfBody {
    std::string name;
    Vec3f pos;                            // Position relative to parent
    Vec4f quat = {0, 0, 0, 1};           // Orientation (x, y, z, w)
    bool has_quat = false;
    std::string childclass;               // Default class for children
    std::vector<MjcfGeom> geoms;          // Collision/visual geometries
    std::vector<MjcfJoint> joints;        // Joint definitions
    std::vector<MjcfBody> children;       // Child bodies
};
```

### MjcfJoint

A joint element parsed from `<joint>`. Supported types: hinge (revolute), slide (prismatic), ball, free.

```cpp
struct MjcfJoint {
    std::string name;
    std::string type = "hinge";           // Joint type
    Vec3f pos;                            // Anchor position
    Vec3f axis = {0, 0, 1};              // Rotation/translation axis
    float range_min = 0;                  // Lower joint limit
    float range_max = 0;                  // Upper joint limit
    bool limited = true;                  // Whether limits are enforced
    float damping = 0.0f;                // Passive damping coefficient
    float stiffness = 0.0f;              // Passive stiffness coefficient
    float armature = 0.0f;               // Rotor inertia
};
```

### MjcfGeom

A collision/visual geometry element parsed from `<geom>`. Supported types: capsule, sphere, box, cylinder, plane.

```cpp
struct MjcfGeom {
    std::string name;
    std::string type = "capsule";
    Vec3f pos;                                         // Local position
    std::array<float, 4> axisangle = {0, 0, 1, 0};    // Axis-angle rotation
    std::optional<std::array<float, 6>> fromto;        // Alternative fromto spec
    std::vector<float> size;                            // Size params (varies by type)
    Vec4f rgba = {0.8f, 0.6f, 0.4f, 1.0f};            // Color
    int condim = 3;                                    // Contact dimensionality
    float friction = 0.4f;                             // Friction coefficient
};
```

### MjcfActuator

An actuator element parsed from `<actuator><motor>`.

```cpp
struct MjcfActuator {
    std::string name;
    std::string joint;            // Name of the joint this actuator drives
    float gear = 1.0f;           // Gear ratio (torque multiplier)
    float ctrl_min = -1.0f;      // Minimum control input
    float ctrl_max = 1.0f;       // Maximum control input
    bool ctrllimited = true;     // Whether control limits are enforced
};
```

### MjcfCompiler

Compiler directives parsed from `<compiler>`.

```cpp
struct MjcfCompiler {
    std::string angle = "degree";        // "degree" or "radian"
    std::string coordinate = "global";   // "global" or "local"
    bool inertiafromgeom = false;        // Compute inertia from geom shapes
    float settotalmass = -1.0f;          // Override total mass (-1 = unset)

    float ToRadians(float val) const;    // Convert based on angle unit
};
```

### MjcfOption

Simulation options parsed from `<option>`.

```cpp
struct MjcfOption {
    Vec3f gravity = {0, 0, -9.81f};     // Gravity vector (m/s^2)
    float timestep = 0.002f;            // Default simulation timestep (seconds)
};
```
