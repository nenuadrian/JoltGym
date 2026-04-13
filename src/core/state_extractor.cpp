#include "state_extractor.h"
#include <cstring>

namespace joltgym {

StateExtractor::StateExtractor(Articulation* articulation, PhysicsWorld* world, int qpos_skip)
    : m_articulation(articulation)
    , m_world(world)
    , m_qpos_skip(qpos_skip)
{
}

int StateExtractor::GetObsDim() const {
    // qpos excluding first `m_qpos_skip` elements + qvel
    return (m_articulation->GetQPosDim() - m_qpos_skip) + m_articulation->GetQVelDim();
}

void StateExtractor::ExtractObs(float* out) const {
    int qpos_dim = m_articulation->GetQPosDim();
    int qvel_dim = m_articulation->GetQVelDim();

    // Temp buffers for qpos and qvel
    std::vector<float> qpos(qpos_dim);
    std::vector<float> qvel(qvel_dim);

    auto& body_interface = m_world->GetPhysicsSystem().GetBodyInterface();
    m_articulation->GetQPos(qpos.data(), body_interface);
    m_articulation->GetQVel(qvel.data(), body_interface);

    // obs = qpos[skip:] ++ qvel
    int idx = 0;
    for (int i = m_qpos_skip; i < qpos_dim; i++) {
        out[idx++] = qpos[i];
    }
    for (int i = 0; i < qvel_dim; i++) {
        out[idx++] = qvel[i];
    }
}

void StateExtractor::ExtractQPos(float* out) const {
    auto& body_interface = m_world->GetPhysicsSystem().GetBodyInterface();
    m_articulation->GetQPos(out, body_interface);
}

void StateExtractor::ExtractQVel(float* out) const {
    auto& body_interface = m_world->GetPhysicsSystem().GetBodyInterface();
    m_articulation->GetQVel(out, body_interface);
}

float StateExtractor::GetRootX() const {
    auto& body_interface = m_world->GetPhysicsSystem().GetBodyInterface();
    return m_articulation->GetRootX(body_interface);
}

float StateExtractor::GetRootXVelocity() const {
    auto& body_interface = m_world->GetPhysicsSystem().GetBodyInterface();
    return m_articulation->GetRootXVelocity(body_interface);
}

float StateExtractor::GetRootZ() const {
    auto& body_interface = m_world->GetPhysicsSystem().GetBodyInterface();
    return m_articulation->GetRootZ(body_interface);
}

} // namespace joltgym
