# Core Physics

The core physics module wraps Jolt Physics and provides articulated body simulation, motor control, and state extraction.

All classes live in the `joltgym` namespace.

## PhysicsWorld

Central simulation container wrapping Jolt's `PhysicsSystem` with snapshot support and articulation management.

Each PhysicsWorld owns a Jolt PhysicsSystem, a body registry, and zero or more Articulation instances. It can operate in shared-thread-pool mode (default) or single-threaded mode (required for WorldPool parallelism).

```cpp
class PhysicsWorld {
public:
    PhysicsWorld();

    /// Initialize the physics system.
    void Init(uint32_t max_bodies = 2048,
              uint32_t max_body_pairs = 4096,
              uint32_t max_contact_constraints = 2048,
              bool single_threaded = false);

    /// Advance physics by dt seconds.
    void Step(float dt, int collision_steps = 1);

    void SetGravity(JPH::Vec3 gravity);

    /// Capture/restore all body states for deterministic reset.
    void SaveSnapshot();
    void RestoreSnapshot();

    /// Add an articulated body to this world.
    void AddArticulation(std::unique_ptr<Articulation> articulation);
    Articulation* GetArticulation(size_t index);
    const std::vector<std::unique_ptr<Articulation>>& GetArticulations() const;

    /// Apply passive damping and stiffness torques for all articulations.
    void ApplyPassiveForces();

    JPH::PhysicsSystem& GetPhysicsSystem();
    JPH::BodyInterface& GetBodyInterface();
    BodyRegistry& GetRegistry();
};
```

| Parameter | Description |
|---|---|
| `max_bodies` | Maximum number of rigid bodies (default: 2048) |
| `max_body_pairs` | Maximum broadphase body pairs (default: 4096) |
| `max_contact_constraints` | Maximum contact constraints (default: 2048) |
| `single_threaded` | If true, uses per-world `JobSystemSingleThreaded` instead of shared thread pool. Required for WorldPool. |

---

## StateSnapshot

Complete snapshot of all body states for deterministic reset.

```cpp
struct StateSnapshot {
    struct BodyState {
        JPH::BodyID id;             // Jolt body identifier
        JPH::RVec3 position;        // World-space position
        JPH::Quat rotation;         // World-space orientation
        JPH::Vec3 linear_velocity;  // Linear velocity (m/s)
        JPH::Vec3 angular_velocity; // Angular velocity (rad/s)
    };
    std::vector<BodyState> bodies;
};
```

---

## Articulation

Articulated body — a tree of rigid bodies connected by motorized joints. Manages the kinematic tree, motor controllers, and state extraction (qpos/qvel) for a single robot. Supports both 2D planar roots (HalfCheetah) and 3D free roots (Humanoid).

```cpp
class Articulation {
public:
    Articulation(const std::string& name);

    void SetRootBody(JPH::BodyID root_body);
    void AddBody(const std::string& name, JPH::BodyID id);
    void AddMotor(std::unique_ptr<MotorController> motor);
    void AddRootDOF(const RootDOF& dof);

    // State dimensions
    int GetQPosDim() const;    // Generalized position coordinates
    int GetQVelDim() const;    // Generalized velocity coordinates
    int GetActionDim() const;  // Number of actuated joints

    // State extraction
    void GetQPos(float* out, const JPH::BodyInterface& body_interface) const;
    void GetQVel(float* out, const JPH::BodyInterface& body_interface) const;

    // Apply normalized actions [-1, 1] to all motor controllers
    void ApplyActions(const float* actions, int count);
    void ApplyPassiveForces(JPH::BodyInterface& body_interface);

    // Root body queries
    float GetRootX(const JPH::BodyInterface& bi) const;
    float GetRootXVelocity(const JPH::BodyInterface& bi) const;
    float GetRootZ(const JPH::BodyInterface& bi) const;
    bool HasFreeRoot() const;
};
```

---

## RootDOF

Describes a single root degree of freedom. Root DOFs map to the position/rotation of the root body and determine the structure of `qpos` and `qvel`.

```cpp
struct RootDOF {
    enum class Type {
        SlideX, SlideZ, HingeY,       // 2D planar (HalfCheetah)
        FreeX, FreeY, FreeZ,          // 3D position (Humanoid)
        QuatW, QuatX, QuatY, QuatZ    // 3D orientation quaternion (Humanoid)
    };
    std::string name;
    Type type;

    bool IsPositionDOF() const;
    static int QPosDim(Type t);
    static bool IsQuatComponent(Type t);
};
```

HalfCheetah uses planar DOFs (`SlideX + SlideZ + HingeY`), while Humanoid uses free DOFs (`FreeX/Y/Z + QuatW/X/Y/Z`). Quaternion DOFs have 4 qpos entries but map to only 3 qvel entries (angular velocity).

---

## MotorController

Single-joint motor controller mapping normalized actions to torques via Jolt's implicit spring integrator.

Maps MuJoCo's motor equation into Jolt:

```
tau = gear * action - stiffness * pos - damping * vel
```

The Jolt position motor uses `StiffnessAndDamping` mode with `target = gear * action / stiffness`, producing stable physics even at high gear ratios because the torque is integrated implicitly.

```cpp
class MotorController {
public:
    enum class JointType { Hinge, Slide };

    MotorController(const std::string& name, JPH::Constraint* constraint,
                    JointType type, float gear_ratio,
                    float ctrl_min, float ctrl_max,
                    float damping, float stiffness, float armature);

    void SetAction(float normalized_action);  // Input in [-1, 1]
    void ApplyPassiveForces(JPH::BodyInterface& body_interface);

    float GetPosition() const;  // Angle (hinge) or displacement (slide)
    float GetVelocity() const;

    const std::string& GetName() const;
    JointType GetType() const;
    float GetGearRatio() const;
    float GetDamping() const;
    float GetStiffness() const;
    float GetLastTorque() const;
};
```

| Parameter | Description |
|---|---|
| `gear_ratio` | Torque multiplier from MJCF actuator gear |
| `ctrl_min/max` | Control input limits |
| `damping` | Passive damping coefficient |
| `stiffness` | Passive stiffness coefficient |
| `armature` | Rotor inertia (added to joint inertia) |

---

## StateExtractor

Extracts observation vectors (`qpos`, `qvel`) from physics state. The observation is `qpos[skip:]` concatenated with `qvel`. The skip parameter removes root position DOFs not useful for policy learning.

```cpp
class StateExtractor {
public:
    /// qpos_skip: 1 for HalfCheetah (skip rootX), 2 for Humanoid (skip rootX, rootY)
    StateExtractor(Articulation* articulation, PhysicsWorld* world, int qpos_skip = 1);

    int GetObsDim() const;     // qpos_dim - skip + qvel_dim
    int GetQPosDim() const;
    int GetQVelDim() const;
    int GetActionDim() const;

    void ExtractObs(float* out) const;   // Full observation: qpos[skip:] ++ qvel
    void ExtractQPos(float* out) const;
    void ExtractQVel(float* out) const;

    float GetRootX() const;
    float GetRootXVelocity() const;
    float GetRootZ() const;
};
```
