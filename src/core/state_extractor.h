#pragma once

#include "articulation.h"
#include "physics_world.h"

namespace joltgym {

class StateExtractor {
public:
    StateExtractor(Articulation* articulation, PhysicsWorld* world);

    int GetObsDim() const;
    int GetQPosDim() const { return m_articulation->GetQPosDim(); }
    int GetQVelDim() const { return m_articulation->GetQVelDim(); }
    int GetActionDim() const { return m_articulation->GetActionDim(); }

    // Extract observation: qpos[1:] ++ qvel (excludes rootx position)
    void ExtractObs(float* out) const;
    void ExtractQPos(float* out) const;
    void ExtractQVel(float* out) const;

    float GetRootX() const;
    float GetRootXVelocity() const;

private:
    Articulation* m_articulation;
    PhysicsWorld* m_world;
};

} // namespace joltgym
