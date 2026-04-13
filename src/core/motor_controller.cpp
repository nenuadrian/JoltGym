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
    // Configure the spring motor once at construction
    if (m_type == JointType::Hinge) {
        auto* hinge = static_cast<JPH::HingeConstraint*>(m_constraint);
        auto& ms = hinge->GetMotorSettings();

        ms.mSpringSettings.mMode = JPH::ESpringMode::StiffnessAndDamping;
        ms.mSpringSettings.mStiffness = m_stiffness;
        ms.mSpringSettings.mDamping = m_damping;

        // High torque limits so the spring is never clipped
        ms.mMaxTorqueLimit = 1000.0f;
        ms.mMinTorqueLimit = -1000.0f;

        // Start in position mode targeting equilibrium (angle 0)
        hinge->SetMotorState(JPH::EMotorState::Position);
        hinge->SetTargetAngle(0.0f);
    }
}

void MotorController::SetAction(float normalized_action) {
    m_pending_action = std::clamp(normalized_action, m_ctrl_min, m_ctrl_max);
}

void MotorController::ApplyPassiveForces(JPH::BodyInterface& body_interface) {
    // Jolt's Position motor with spring applies:
    //   tau = -stiffness * (pos - target) - damping * vel
    //
    // MuJoCo's joint forces:
    //   tau = gear * action - stiffness * pos - damping * vel
    //
    // Setting target = gear * action / stiffness gives:
    //   tau = -stiffness * (pos - gear*action/stiffness) - damping * vel
    //       = -stiffness * pos + gear * action - damping * vel  ✓

    float actuator_torque = m_pending_action * m_gear_ratio;
    m_last_torque = actuator_torque;

    if (m_type == JointType::Hinge) {
        auto* hinge = static_cast<JPH::HingeConstraint*>(m_constraint);

        if (m_stiffness > 1e-6f) {
            // Offset target angle to inject actuator torque through the spring
            float target = actuator_torque / m_stiffness;
            hinge->SetTargetAngle(target);
        } else {
            // No stiffness — use velocity motor for actuation + damping
            auto& ms = hinge->GetMotorSettings();
            float total = actuator_torque - m_damping * GetVelocity();
            if (std::abs(total) < 1e-8f) {
                hinge->SetMotorState(JPH::EMotorState::Off);
            } else {
                ms.mSpringSettings.mMode = JPH::ESpringMode::StiffnessAndDamping;
                ms.mSpringSettings.mStiffness = 0;
                ms.mSpringSettings.mDamping = 0;
                ms.mMaxTorqueLimit = std::abs(total);
                ms.mMinTorqueLimit = -std::abs(total);
                hinge->SetMotorState(JPH::EMotorState::Velocity);
                hinge->SetTargetAngularVelocity(total > 0 ? 1000.0f : -1000.0f);
            }
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
