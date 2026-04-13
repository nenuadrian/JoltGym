#pragma once

#include "articulation.h"
#include "physics_world.h"

namespace joltgym {

class StateExtractor {
public:
    // qpos_skip: number of leading qpos elements to exclude from obs
    // Default 1 (skip rootx) for HalfCheetah. Use 2 for Humanoid (skip x,y).
    StateExtractor(Articulation* articulation, PhysicsWorld* world, int qpos_skip = 1);

    int GetObsDim() const;
    int GetQPosDim() const { return m_articulation->GetQPosDim(); }
    int GetQVelDim() const { return m_articulation->GetQVelDim(); }
    int GetActionDim() const { return m_articulation->GetActionDim(); }

    // Extract observation: qpos[skip:] ++ qvel
    void ExtractObs(float* out) const;
    void ExtractQPos(float* out) const;
    void ExtractQVel(float* out) const;

    float GetRootX() const;
    float GetRootXVelocity() const;
    float GetRootZ() const;

private:
    Articulation* m_articulation;
    PhysicsWorld* m_world;
    int m_qpos_skip;
};

} // namespace joltgym
