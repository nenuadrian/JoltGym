# MJCF Parser

The MJCF module parses MuJoCo XML model files and converts them into Jolt Physics bodies and constraints.

## Overview

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

## Data Model

`src/mjcf/mjcf_model.h`

### MjcfModel

The top-level parsed representation of a MuJoCo XML file:

```cpp
struct MjcfModel {
    std::string name;
    MjcfCompiler compiler;   // angle units, coordinate frame
    MjcfOption option;       // gravity, timestep
    MjcfBody worldbody;      // root of body tree
    std::vector<MjcfActuator> actuators;
};
```

### MjcfBody

Hierarchical body definition:

```cpp
struct MjcfBody {
    std::string name;
    Vec3f pos;                     // position relative to parent
    Vec4f quat;                    // orientation (x, y, z, w)
    bool has_quat;
    std::string childclass;        // default class name
    std::vector<MjcfGeom> geoms;   // collision/visual geometries
    std::vector<MjcfJoint> joints; // joint definitions
    std::vector<MjcfBody> children; // child bodies
};
```

### MjcfJoint

Joint definition supporting hinge, slide, ball, and free types:

```cpp
struct MjcfJoint {
    std::string name;
    std::string type;      // "hinge", "slide", "ball", "free"
    Vec3f pos;             // joint anchor position
    Vec3f axis;            // rotation/translation axis
    float range_min, range_max;
    bool limited;
    float damping;
    float stiffness;
    float armature;
};
```

### MjcfGeom

Collision and visual geometry:

```cpp
struct MjcfGeom {
    std::string name;
    std::string type;       // "capsule", "sphere", "box", "cylinder", "plane"
    Vec3f pos;
    std::array<float, 4> axisangle;
    std::optional<std::array<float, 6>> fromto;  // alternative to pos+size
    std::vector<float> size;
    Vec4f rgba;
    int condim;
    float friction;
};
```

### MjcfActuator

Motor definition linking to a named joint:

```cpp
struct MjcfActuator {
    std::string name;
    std::string joint;     // name of the joint to actuate
    float gear;            // gear ratio (torque multiplier)
    float ctrl_min, ctrl_max;
    bool ctrllimited;
};
```

### MjcfCompiler

Metadata controlling how the model is interpreted:

```cpp
struct MjcfCompiler {
    std::string angle;      // "degree" or "radian"
    std::string coordinate; // "global" or "local"
    bool inertiafromgeom;
    float settotalmass;     // -1 = unset

    float ToRadians(float val) const;
};
```

## Parser

`src/mjcf/mjcf_parser.h`

```cpp
class MjcfParser {
public:
    static MjcfModel Parse(const std::string& xml_path);
};
```

Parses a MuJoCo XML file using `tinyxml2`. Handles:

- Body hierarchy traversal
- Geom/joint/actuator parsing
- Default class inheritance
- Compiler directives (angle units, coordinate frames)
- `fromto` syntax for capsule geoms

## MJCF to Jolt Conversion

`src/mjcf/mjcf_to_jolt.h`

```cpp
class MjcfToJolt {
public:
    static void Build(const MjcfModel& model,
                      PhysicsWorld& world,
                      Articulation& articulation);
};
```

Converts the parsed model into Jolt Physics objects:

- Creates `Body` instances for each `MjcfBody`
- Creates `HingeConstraint` or `SliderConstraint` for each `MjcfJoint`
- Sets up `MotorController` for each `MjcfActuator`
- Configures collision shapes from `MjcfGeom` data

### Multi-Joint Decomposition

Bodies with multiple hinge joints (e.g., humanoid hip with 3 DOFs) are decomposed into a chain of intermediate bodies:

```
parent -> hinge_1 -> intermediate_1 -> hinge_2 -> intermediate_2 -> hinge_3 -> actual_body
```

Intermediate bodies are tiny-mass spheres (0.001 kg) that act purely as kinematic connectors. This is necessary because Jolt's constraint system connects exactly two bodies, so a 3-DOF joint requires 2 intermediate bodies and 3 hinge constraints.

## Supported MJCF Features

| Feature | Status |
|---|---|
| Body hierarchy | Supported |
| Hinge joints | Supported |
| Slide joints | Supported |
| Free joints (6DOF) | Supported |
| Ball joints | Parsed, converted to 3 hinges |
| Capsule geoms | Supported |
| Sphere geoms | Supported |
| Box geoms | Supported |
| Cylinder geoms | Supported |
| Plane geoms | Supported |
| Actuators (motor) | Supported |
| Default classes | Supported |
| Compiler directives | Supported |
| `fromto` syntax | Supported |
| Tendons | Parsed (not physically simulated) |
| Contacts/exclude | Not yet supported |
| Sensors | Not yet supported |
| Equality constraints | Not yet supported |

## Default Classes

`src/mjcf/mjcf_defaults.h`

MuJoCo's default system allows geoms and joints to inherit properties from named classes:

```xml
<default>
  <default class="body">
    <geom type="capsule" size="0.046"/>
    <joint damping="0.5"/>
  </default>
</default>
```

The parser resolves defaults before conversion, merging class properties into each element.
