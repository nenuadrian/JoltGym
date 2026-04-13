#pragma once

#include "joltgym_core.h"
#include "motor_controller.h"
#include <string>
#include <vector>
#include <memory>

namespace joltgym {

// Describes a root DOF that maps to position/rotation of the root body
struct RootDOF {
    enum class Type {
        SlideX, SlideZ, HingeY,                        // 2D planar (HalfCheetah)
        FreeX, FreeY, FreeZ,                            // 3D position
        QuatW, QuatX, QuatY, QuatZ,                     // 3D orientation (quaternion)
    };
    std::string name;
    Type type;

    // Is this a position DOF (contributes to qpos)?
    bool IsPositionDOF() const {
        return type != Type::QuatW; // All except QuatW are directly in qpos
        // Actually all root DOF types contribute to qpos
    }

    // How many qpos entries does this DOF add?
    static int QPosDim(Type t) { return 1; }

    // Is this DOF velocity-only (no qpos counterpart)?
    // QuatW/X/Y/Z have 4 qpos entries but only 3 qvel entries (angular velocity)
    static bool IsQuatComponent(Type t) {
        return t == Type::QuatW || t == Type::QuatX ||
               t == Type::QuatY || t == Type::QuatZ;
    }
};

class Articulation {
public:
    Articulation(const std::string& name);

    void SetRootBody(JPH::BodyID root_body);
    void AddBody(const std::string& name, JPH::BodyID id);
    void AddMotor(std::unique_ptr<MotorController> motor);
    void AddRootDOF(const RootDOF& dof);

    // State dimensions
    int GetQPosDim() const;
    int GetQVelDim() const;
    int GetActionDim() const;

    // Extract state vectors
    void GetQPos(float* out, const JPH::BodyInterface& body_interface) const;
    void GetQVel(float* out, const JPH::BodyInterface& body_interface) const;

    // Apply actions to motors
    void ApplyActions(const float* actions, int count);
    void ApplyPassiveForces(JPH::BodyInterface& body_interface);

    // Access
    const std::string& GetName() const { return m_name; }
    JPH::BodyID GetRootBody() const { return m_root_body; }
    const std::vector<JPH::BodyID>& GetBodies() const { return m_bodies; }
    const std::vector<std::string>& GetBodyNames() const { return m_body_names; }
    const std::vector<std::unique_ptr<MotorController>>& GetMotors() const { return m_motors; }
    MotorController* GetMotor(size_t index) { return m_motors[index].get(); }

    float GetRootX(const JPH::BodyInterface& body_interface) const;
    float GetRootXVelocity(const JPH::BodyInterface& body_interface) const;
    float GetRootZ(const JPH::BodyInterface& body_interface) const;

    // Does this articulation have a free (6DOF) root?
    bool HasFreeRoot() const { return m_has_free_root; }
    void SetHasFreeRoot(bool v) { m_has_free_root = v; }

private:
    bool m_has_free_root = false;
    std::string m_name;
    JPH::BodyID m_root_body;
    std::vector<std::string> m_body_names;
    std::vector<JPH::BodyID> m_bodies;
    std::vector<std::unique_ptr<MotorController>> m_motors;
    std::vector<RootDOF> m_root_dofs;
};

} // namespace joltgym
