/// @file state_extractor.h
/// @brief Extracts observation vectors (qpos, qvel) from physics state.
#pragma once

#include "articulation.h"
#include "physics_world.h"

namespace joltgym {

/// @brief Extracts observation vectors from the physics state of an Articulation.
///
/// The observation is `qpos[skip:]` concatenated with `qvel`. The skip
/// parameter removes root position DOFs that are not useful for policy
/// learning (e.g., root X for HalfCheetah, root X+Y for Humanoid).
class StateExtractor {
public:
    /// @param articulation The articulated body to observe.
    /// @param world        The physics world containing the body.
    /// @param qpos_skip    Number of leading qpos elements to exclude.
    ///                     Default 1 (skip rootX) for HalfCheetah.
    ///                     Use 2 for Humanoid (skip rootX, rootY).
    StateExtractor(Articulation* articulation, PhysicsWorld* world, int qpos_skip = 1);

    /// @brief Total observation dimension: `qpos_dim - skip + qvel_dim`.
    int GetObsDim() const;

    int GetQPosDim() const { return m_articulation->GetQPosDim(); }  ///< Full qpos dimension.
    int GetQVelDim() const { return m_articulation->GetQVelDim(); }  ///< Full qvel dimension.
    int GetActionDim() const { return m_articulation->GetActionDim(); } ///< Number of actuated joints.

    /// @brief Extract the full observation vector: `qpos[skip:] ++ qvel`.
    void ExtractObs(float* out) const;

    /// @brief Extract full generalized positions.
    void ExtractQPos(float* out) const;

    /// @brief Extract full generalized velocities.
    void ExtractQVel(float* out) const;

    float GetRootX() const;         ///< Root body X position.
    float GetRootXVelocity() const; ///< Root body X velocity.
    float GetRootZ() const;         ///< Root body Z position (height).

private:
    Articulation* m_articulation;
    PhysicsWorld* m_world;
    int m_qpos_skip;
};

} // namespace joltgym
