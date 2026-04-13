# Core Physics

The core physics module wraps Jolt Physics and provides articulated body simulation, motor control, and state extraction.

## PhysicsWorld

`src/core/physics_world.h`

The central simulation container. Wraps Jolt's `PhysicsSystem` with snapshot support and articulation management.

### Class Definition

```cpp
class PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld();

    void Init(uint32_t max_bodies = 2048,
              uint32_t max_body_pairs = 4096,
              uint32_t max_contact_constraints = 2048,
              bool single_threaded = false);

    void Step(float dt, int collision_steps = 1);
    void SetGravity(JPH::Vec3 gravity);

    // State snapshots for deterministic reset
    void SaveSnapshot();
    void RestoreSnapshot();

    // Articulation management
    void AddArticulation(std::unique_ptr<Articulation> articulation);
    Articulation* GetArticulation(size_t index);
    const std::vector<std::unique_ptr<Articulation>>& GetArticulations() const;

    // Apply passive damping/stiffness for all articulations
    void ApplyPassiveForces();

    // Direct access
    JPH::PhysicsSystem& GetPhysicsSystem();
    JPH::BodyInterface& GetBodyInterface();
    BodyRegistry& GetRegistry();
};
```

### State Snapshots

`SaveSnapshot()` captures the position, rotation, and velocity of every body. `RestoreSnapshot()` restores them exactly, enabling deterministic resets.

```cpp
struct StateSnapshot {
    struct BodyState {
        JPH::BodyID id;
        JPH::RVec3 position;
        JPH::Quat rotation;
        JPH::Vec3 linear_velocity;
        JPH::Vec3 angular_velocity;
    };
    std::vector<BodyState> bodies;
};
```

### Threading Modes

| Mode | When to Use |
|---|---|
| `single_threaded = false` | Default. Uses the shared Jolt thread pool. Best for single-world usage. |
| `single_threaded = true` | Uses `JobSystemSingleThreaded`. Required for `WorldPool` to avoid contention. |

---

## Articulation

`src/core/articulation.h`

Represents an articulated body -- a tree of rigid bodies connected by joints with motor controllers.

### Class Definition

```cpp
class Articulation {
public:
    Articulation(const std::string& name);

    void SetRootBody(JPH::BodyID root_body);
    void AddBody(const std::string& name, JPH::BodyID id);
    void AddMotor(std::unique_ptr<MotorController> motor);
    void AddRootDOF(const RootDOF& dof);

    // State dimensions
    int GetQPosDim() const;   // Number of position DOFs
    int GetQVelDim() const;   // Number of velocity DOFs
    int GetActionDim() const; // Number of actuated joints

    // State extraction
    void GetQPos(float* out, const JPH::BodyInterface& bi) const;
    void GetQVel(float* out, const JPH::BodyInterface& bi) const;

    // Control
    void ApplyActions(const float* actions, int count);
    void ApplyPassiveForces(JPH::BodyInterface& bi);

    // Root body queries
    float GetRootX(const JPH::BodyInterface& bi) const;
    float GetRootZ(const JPH::BodyInterface& bi) const;
    float GetRootXVelocity(const JPH::BodyInterface& bi) const;

    bool HasFreeRoot() const;
};
```

### Root DOF Types

The root body's degrees of freedom are configured via `RootDOF`:

```cpp
enum class RootDOF::Type {
    SlideX, SlideZ, HingeY,    // 2D planar (HalfCheetah)
    FreeX, FreeY, FreeZ,       // 3D position (Humanoid)
    QuatW, QuatX, QuatY, QuatZ // 3D orientation (Humanoid)
};
```

**HalfCheetah** uses `SlideX + SlideZ + HingeY` (3 root DOFs).

**Humanoid** uses `FreeX + FreeY + FreeZ + QuatW + QuatX + QuatY + QuatZ` (7 root qpos, 6 root qvel).

---

## MotorController

`src/core/motor_controller.h`

Controls a single joint. Maps normalized actions to torques and applies passive forces.

### Class Definition

```cpp
class MotorController {
public:
    enum class JointType { Hinge, Slide };

    MotorController(const std::string& name,
                    JPH::Constraint* constraint,
                    JointType type,
                    float gear_ratio,
                    float ctrl_min, float ctrl_max,
                    float damping, float stiffness,
                    float armature);

    void SetAction(float normalized_action);  // [-1, 1]
    void ApplyPassiveForces(JPH::BodyInterface& bi);

    float GetPosition() const;
    float GetVelocity() const;
    float GetLastTorque() const;

    // Properties
    const std::string& GetName() const;
    JointType GetType() const;
    float GetGearRatio() const;
    float GetDamping() const;
    float GetStiffness() const;
};
```

### Motor Model

The motor maps MuJoCo's force equation into Jolt's implicit spring integrator:

$$
\tau = \text{gear} \times \text{action} - \text{stiffness} \times \text{pos} - \text{damping} \times \text{vel}
$$

Jolt's position motor uses `StiffnessAndDamping` mode. The target position is set to:

$$
\text{target} = \frac{\text{gear} \times \text{action}}{\text{stiffness}}
$$

This produces stable physics even at high gear ratios, because the torque is integrated implicitly rather than applied explicitly.

---

## StateExtractor

`src/core/state_extractor.h`

Extracts observation vectors from the physics state.

```cpp
class StateExtractor {
public:
    // qpos_skip: elements to skip from qpos (1 for HalfCheetah, 2 for Humanoid)
    StateExtractor(Articulation* articulation, PhysicsWorld* world,
                   int qpos_skip = 1);

    int GetObsDim() const;    // qpos_dim - skip + qvel_dim
    int GetActionDim() const;

    void ExtractObs(float* out) const;   // qpos[skip:] ++ qvel
    void ExtractQPos(float* out) const;
    void ExtractQVel(float* out) const;

    float GetRootX() const;
    float GetRootXVelocity() const;
    float GetRootZ() const;
};
```

The observation vector is `qpos[skip:]` concatenated with `qvel`. The skip parameter removes the root X position (HalfCheetah) or root X and Y positions (Humanoid) from the observation, as these are not useful for the policy.

---

## Collision Layers

`src/core/collision_layers.h`

Configures broadphase collision filtering. Used by the multi-agent environment to control which bodies can collide.

---

## Body Registry

`src/core/body_registry.h`

Maintains a name-to-`BodyID` mapping so bodies can be looked up by their MJCF name (e.g., `"torso"`, `"bthigh"`).
