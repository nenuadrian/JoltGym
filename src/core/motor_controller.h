#pragma once

#include "joltgym_core.h"
#include <string>

namespace joltgym {

class MotorController {
public:
    enum class JointType { Hinge, Slide };

    MotorController(const std::string& name, JPH::Constraint* constraint,
                    JointType type, float gear_ratio,
                    float ctrl_min, float ctrl_max,
                    float damping, float stiffness, float armature);

    // Apply normalized action [-1, 1] -> torque via gear ratio
    void SetAction(float normalized_action);

    // Apply passive damping and stiffness torques
    void ApplyPassiveForces(JPH::BodyInterface& body_interface);

    // Read joint state
    float GetPosition() const;
    float GetVelocity() const;

    const std::string& GetName() const { return m_name; }
    JointType GetType() const { return m_type; }
    float GetGearRatio() const { return m_gear_ratio; }
    float GetDamping() const { return m_damping; }
    float GetStiffness() const { return m_stiffness; }
    float GetLastTorque() const { return m_last_torque; }

    JPH::Constraint* GetConstraint() { return m_constraint; }

private:
    std::string m_name;
    JPH::Constraint* m_constraint;
    JointType m_type;
    float m_gear_ratio;
    float m_ctrl_min;
    float m_ctrl_max;
    float m_damping;
    float m_stiffness;
    float m_armature;
    float m_last_torque = 0.0f;
};

} // namespace joltgym
