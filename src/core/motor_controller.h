/// @file motor_controller.h
/// @brief Single-joint motor controller mapping normalized actions to torques.
#pragma once

#include "joltgym_core.h"
#include <string>

namespace joltgym {

/// @brief Controls a single actuated joint.
///
/// Maps MuJoCo's motor equation into Jolt's implicit spring integrator:
///
///     tau = gear * action - stiffness * pos - damping * vel
///
/// The Jolt position motor uses `StiffnessAndDamping` mode with:
///
///     target = gear * action / stiffness
///
/// This produces stable physics even at high gear ratios because the torque
/// is integrated implicitly rather than applied explicitly.
class MotorController {
public:
    /// @brief Joint type.
    enum class JointType {
        Hinge, ///< Revolute (rotation) joint.
        Slide  ///< Prismatic (translation) joint.
    };

    /// @param name       Joint name from the MJCF model.
    /// @param constraint  Jolt constraint this motor acts on.
    /// @param type       Hinge or Slide.
    /// @param gear_ratio Torque multiplier (from MJCF actuator gear).
    /// @param ctrl_min   Minimum control input.
    /// @param ctrl_max   Maximum control input.
    /// @param damping    Passive damping coefficient.
    /// @param stiffness  Passive stiffness coefficient.
    /// @param armature   Rotor inertia (added to joint inertia).
    MotorController(const std::string& name, JPH::Constraint* constraint,
                    JointType type, float gear_ratio,
                    float ctrl_min, float ctrl_max,
                    float damping, float stiffness, float armature);

    /// @brief Apply a normalized action in [-1, 1], scaled by gear ratio.
    void SetAction(float normalized_action);

    /// @brief Apply passive damping and stiffness torques to the joint.
    void ApplyPassiveForces(JPH::BodyInterface& body_interface);

    /// @brief Current joint position (angle for hinge, displacement for slide).
    float GetPosition() const;

    /// @brief Current joint velocity.
    float GetVelocity() const;

    /// @name Properties
    /// @{
    const std::string& GetName() const { return m_name; }
    JointType GetType() const { return m_type; }
    float GetGearRatio() const { return m_gear_ratio; }
    float GetDamping() const { return m_damping; }
    float GetStiffness() const { return m_stiffness; }
    float GetLastTorque() const { return m_last_torque; } ///< Torque applied on the last step.
    JPH::Constraint* GetConstraint() { return m_constraint; }
    /// @}

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
    float m_pending_action = 0.0f;
};

} // namespace joltgym
