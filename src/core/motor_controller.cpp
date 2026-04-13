#include "motor_controller.h"
#include <cmath>
#include <algorithm>

namespace joltgym {

MotorController::MotorController(const std::string& name, JPH::Constraint* constraint,
                                 JointType type, float gear_ratio,
                                 float ctrl_min, float ctrl_max,
                                 float damping, float stiffness, float armature)
    : m_name(name)
    , m_constraint(constraint)
    , m_type(type)
    , m_gear_ratio(gear_ratio)
    , m_ctrl_min(ctrl_min)
    , m_ctrl_max(ctrl_max)
    , m_damping(damping)
    , m_stiffness(stiffness)
    , m_armature(armature)
{
    // Configure motor based on stiffness mode
    if (m_type == JointType::Hinge) {
        auto* hinge = static_cast<JPH::HingeConstraint*>(m_constraint);
        auto& ms = hinge->GetMotorSettings();

        // High torque limits so the spring/damper is never clipped
        ms.mMaxTorqueLimit = 5000.0f;
        ms.mMinTorqueLimit = -5000.0f;

        ms.mSpringSettings.mMode = JPH::ESpringMode::StiffnessAndDamping;

        if (m_stiffness > 1e-6f) {
            // Position motor with spring: tau = -stiffness*(pos-target) - damping*vel
            ms.mSpringSettings.mStiffness = m_stiffness;
            ms.mSpringSettings.mDamping = m_damping;
            hinge->SetMotorState(JPH::EMotorState::Position);
            hinge->SetTargetAngle(0.0f);
        } else {
            // Zero-stiffness joint: velocity motor for damping + actuation
            // tau = -damping * (vel - target_vel)
            // With target_vel = gear*action/damping → tau = -damping*vel + gear*action
            ms.mSpringSettings.mStiffness = 0.0f;
            ms.mSpringSettings.mDamping = std::max(m_damping, 0.1f); // Ensure some damping
            hinge->SetMotorState(JPH::EMotorState::Velocity);
            hinge->SetTargetAngularVelocity(0.0f);
        }
    }
}

void MotorController::SetAction(float normalized_action) {
    m_pending_action = std::clamp(normalized_action, m_ctrl_min, m_ctrl_max);
}

void MotorController::ApplyPassiveForces(JPH::BodyInterface& body_interface) {
    // MuJoCo joint force model:
    //   tau = gear * action - stiffness * pos - damping * vel
    //
    // For position motor (stiffness > 0):
    //   Set target = gear * action / stiffness:
    //   tau = -stiffness * (pos - gear*action/stiffness) - damping * vel
    //       = gear * action - stiffness * pos - damping * vel  ✓
    //
    // For velocity motor (stiffness = 0):
    //   Set target_vel = gear * action / damping:
    //   tau = -damping * (vel - gear*action/damping)
    //       = gear * action - damping * vel  ✓

    float actuator_torque = m_pending_action * m_gear_ratio;
    m_last_torque = actuator_torque;

    if (m_type == JointType::Hinge) {
        auto* hinge = static_cast<JPH::HingeConstraint*>(m_constraint);

        if (m_stiffness > 1e-6f) {
            // Position motor: offset target angle to inject actuator torque
            float target = actuator_torque / m_stiffness;
            hinge->SetTargetAngle(target);
        } else {
            // Velocity motor: set target velocity to inject actuator torque
            float eff_damping = std::max(m_damping, 0.1f);
            float target_vel = actuator_torque / eff_damping;
            hinge->SetTargetAngularVelocity(target_vel);
        }
    }
}

float MotorController::GetPosition() const {
    if (m_type == JointType::Hinge) {
        auto* hinge = static_cast<JPH::HingeConstraint*>(m_constraint);
        return hinge->GetCurrentAngle();
    } else {
        auto* slider = static_cast<JPH::SliderConstraint*>(m_constraint);
        return slider->GetCurrentPosition();
    }
}

float MotorController::GetVelocity() const {
    if (m_type == JointType::Hinge) {
        auto* hinge = static_cast<JPH::HingeConstraint*>(m_constraint);
        JPH::Vec3 local_axis = hinge->GetLocalSpaceHingeAxis2();
        JPH::Vec3 axis = hinge->GetBody2()->GetRotation() * local_axis;
        JPH::Vec3 omega1 = hinge->GetBody1()->GetAngularVelocity();
        JPH::Vec3 omega2 = hinge->GetBody2()->GetAngularVelocity();
        return (omega2 - omega1).Dot(axis);
    } else {
        auto* slider = static_cast<JPH::SliderConstraint*>(m_constraint);
        JPH::Vec3 local_axis(1, 0, 0);
        JPH::Vec3 axis = slider->GetBody1()->GetRotation() * local_axis;
        JPH::Vec3 v1 = slider->GetBody1()->GetLinearVelocity();
        JPH::Vec3 v2 = slider->GetBody2()->GetLinearVelocity();
        return (v2 - v1).Dot(axis);
    }
}

} // namespace joltgym
