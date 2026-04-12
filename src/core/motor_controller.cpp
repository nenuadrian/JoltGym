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
}

void MotorController::SetAction(float normalized_action) {
    float clamped = std::clamp(normalized_action, m_ctrl_min, m_ctrl_max);
    float torque = clamped * m_gear_ratio;
    m_last_torque = torque;

    if (m_type == JointType::Hinge) {
        auto* hinge = static_cast<JPH::HingeConstraint*>(m_constraint);
        // Use velocity motor with torque cap to simulate direct torque actuator
        auto& motor_settings = hinge->GetMotorSettings();
        motor_settings.mMaxTorqueLimit = std::abs(torque);
        motor_settings.mMinTorqueLimit = -std::abs(torque);
        hinge->SetMotorState(JPH::EMotorState::Velocity);
        // Large target velocity in direction of desired torque
        hinge->SetTargetAngularVelocity(torque > 0 ? 100.0f : -100.0f);
    } else if (m_type == JointType::Slide) {
        auto* slider = static_cast<JPH::SliderConstraint*>(m_constraint);
        auto& motor_settings = slider->GetMotorSettings();
        motor_settings.mMaxForceLimit = std::abs(torque);
        motor_settings.mMinForceLimit = -std::abs(torque);
        slider->SetMotorState(JPH::EMotorState::Velocity);
        slider->SetTargetVelocity(torque > 0 ? 100.0f : -100.0f);
    }
}

void MotorController::ApplyPassiveForces(JPH::BodyInterface& body_interface) {
    // Apply damping and stiffness as explicit torques
    if (m_damping <= 0.0f && m_stiffness <= 0.0f) return;

    float vel = GetVelocity();
    float pos = GetPosition();

    float passive_torque = 0.0f;
    if (m_damping > 0.0f)   passive_torque -= m_damping * vel;
    if (m_stiffness > 0.0f) passive_torque -= m_stiffness * pos;

    if (std::abs(passive_torque) < 1e-8f) return;

    if (m_type == JointType::Hinge) {
        auto* hinge = static_cast<JPH::HingeConstraint*>(m_constraint);
        // Compute world-space axis from local-space axis and body rotation
        JPH::Vec3 local_axis = hinge->GetLocalSpaceHingeAxis2();
        JPH::Vec3 axis = hinge->GetBody2()->GetRotation() * local_axis;
        JPH::BodyID body2_id = hinge->GetBody2()->GetID();
        body_interface.AddTorque(body2_id, axis * passive_torque);
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
        JPH::Vec3 local_axis(1, 0, 0); // Default slider axis
        JPH::Vec3 axis = slider->GetBody1()->GetRotation() * local_axis;
        JPH::Vec3 v1 = slider->GetBody1()->GetLinearVelocity();
        JPH::Vec3 v2 = slider->GetBody2()->GetLinearVelocity();
        return (v2 - v1).Dot(axis);
    }
}

} // namespace joltgym
