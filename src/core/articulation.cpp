#include "articulation.h"
#include <cmath>
#include <cassert>

namespace joltgym {

Articulation::Articulation(const std::string& name)
    : m_name(name)
{
}

void Articulation::SetRootBody(JPH::BodyID root_body) {
    m_root_body = root_body;
}

void Articulation::AddBody(const std::string& name, JPH::BodyID id) {
    m_body_names.push_back(name);
    m_bodies.push_back(id);
}

void Articulation::AddMotor(std::unique_ptr<MotorController> motor) {
    m_motors.push_back(std::move(motor));
}

void Articulation::AddRootDOF(const RootDOF& dof) {
    m_root_dofs.push_back(dof);
}

int Articulation::GetQPosDim() const {
    return (int)m_root_dofs.size() + (int)m_motors.size();
}

int Articulation::GetQVelDim() const {
    return GetQPosDim(); // Same dimensionality for hinge+slide DOFs
}

int Articulation::GetActionDim() const {
    return (int)m_motors.size();
}

void Articulation::GetQPos(float* out, const JPH::BodyInterface& body_interface) const {
    int idx = 0;

    // Root DOFs
    JPH::RVec3 root_pos = body_interface.GetPosition(m_root_body);
    JPH::Quat root_rot = body_interface.GetRotation(m_root_body);

    for (auto& dof : m_root_dofs) {
        switch (dof.type) {
            case RootDOF::Type::SlideX:
                out[idx++] = (float)root_pos.GetX();
                break;
            case RootDOF::Type::SlideZ:
                out[idx++] = (float)root_pos.GetZ();
                break;
            case RootDOF::Type::HingeY: {
                // Extract Y-axis rotation from quaternion (for 2D planar robot)
                // Using atan2 for the rotation around Y axis
                float siny_cosp = 2.0f * (root_rot.GetW() * root_rot.GetY()
                                         - root_rot.GetX() * root_rot.GetZ());
                float cosy_cosp = 1.0f - 2.0f * (root_rot.GetY() * root_rot.GetY()
                                                 + root_rot.GetZ() * root_rot.GetZ());
                out[idx++] = std::atan2(siny_cosp, cosy_cosp);
                break;
            }
        }
    }

    // Joint DOFs (motors)
    for (auto& motor : m_motors) {
        out[idx++] = motor->GetPosition();
    }
}

void Articulation::GetQVel(float* out, const JPH::BodyInterface& body_interface) const {
    int idx = 0;

    // Root DOF velocities
    JPH::Vec3 root_lin_vel = body_interface.GetLinearVelocity(m_root_body);
    JPH::Vec3 root_ang_vel = body_interface.GetAngularVelocity(m_root_body);

    for (auto& dof : m_root_dofs) {
        switch (dof.type) {
            case RootDOF::Type::SlideX:
                out[idx++] = root_lin_vel.GetX();
                break;
            case RootDOF::Type::SlideZ:
                out[idx++] = root_lin_vel.GetZ();
                break;
            case RootDOF::Type::HingeY:
                out[idx++] = root_ang_vel.GetY();
                break;
        }
    }

    // Joint DOF velocities
    for (auto& motor : m_motors) {
        out[idx++] = motor->GetVelocity();
    }
}

void Articulation::ApplyActions(const float* actions, int count) {
    assert(count == (int)m_motors.size());
    for (int i = 0; i < count; i++) {
        m_motors[i]->SetAction(actions[i]);
    }
}

void Articulation::ApplyPassiveForces(JPH::BodyInterface& body_interface) {
    for (auto& motor : m_motors) {
        motor->ApplyPassiveForces(body_interface);
    }
}

float Articulation::GetRootX(const JPH::BodyInterface& body_interface) const {
    return (float)body_interface.GetPosition(m_root_body).GetX();
}

float Articulation::GetRootXVelocity(const JPH::BodyInterface& body_interface) const {
    return body_interface.GetLinearVelocity(m_root_body).GetX();
}

} // namespace joltgym
