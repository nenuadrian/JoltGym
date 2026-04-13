/// @file physics_world.h
/// @brief Central physics simulation container wrapping Jolt PhysicsSystem.
#pragma once

#include "joltgym_core.h"
#include "collision_layers.h"
#include "body_registry.h"

#include <vector>
#include <memory>

namespace joltgym {

class Articulation;

/// @brief Complete snapshot of all body states for deterministic reset.
struct StateSnapshot {
    /// @brief Position, rotation, and velocity of a single body.
    struct BodyState {
        JPH::BodyID id;                ///< Jolt body identifier.
        JPH::RVec3 position;           ///< World-space position.
        JPH::Quat rotation;            ///< World-space orientation.
        JPH::Vec3 linear_velocity;     ///< Linear velocity (m/s).
        JPH::Vec3 angular_velocity;    ///< Angular velocity (rad/s).
    };
    std::vector<BodyState> bodies;     ///< All captured body states.
};

/// @brief Central physics world — wraps Jolt's PhysicsSystem with snapshot
///        support and articulation management.
///
/// Each PhysicsWorld owns a Jolt PhysicsSystem, a body registry, and zero or
/// more Articulation instances. It can operate in shared-thread-pool mode
/// (default) or single-threaded mode (required for WorldPool parallelism).
class PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld();

    /// @brief Initialize the physics system.
    /// @param max_bodies          Maximum number of rigid bodies.
    /// @param max_body_pairs      Maximum broadphase body pairs.
    /// @param max_contact_constraints Maximum contact constraints.
    /// @param single_threaded     If true, uses a per-world JobSystemSingleThreaded
    ///                            instead of the shared thread pool. Required for
    ///                            WorldPool to avoid contention.
    void Init(uint32_t max_bodies = 2048, uint32_t max_body_pairs = 4096,
              uint32_t max_contact_constraints = 2048,
              bool single_threaded = false);

    /// @brief Advance physics by @p dt seconds.
    /// @param dt              Time step in seconds.
    /// @param collision_steps Number of collision sub-steps.
    void Step(float dt, int collision_steps = 1);

    /// @brief Set the gravity vector.
    void SetGravity(JPH::Vec3 gravity);

    /// @brief Capture the current state of all bodies for later restore.
    void SaveSnapshot();

    /// @brief Restore all bodies to the last saved snapshot.
    void RestoreSnapshot();

    /// @name Accessors
    /// @{
    JPH::PhysicsSystem& GetPhysicsSystem() { return m_physics_system; }
    const JPH::PhysicsSystem& GetPhysicsSystem() const { return m_physics_system; }
    JPH::BodyInterface& GetBodyInterface() { return m_physics_system.GetBodyInterface(); }
    BodyRegistry& GetRegistry() { return m_registry; }
    const BodyRegistry& GetRegistry() const { return m_registry; }
    /// @}

    /// @brief Add an articulated body to this world.
    void AddArticulation(std::unique_ptr<Articulation> articulation);

    /// @brief Get all articulations.
    const std::vector<std::unique_ptr<Articulation>>& GetArticulations() const { return m_articulations; }

    /// @brief Get a specific articulation by index.
    Articulation* GetArticulation(size_t index) { return m_articulations[index].get(); }

    /// @brief Apply passive damping and stiffness torques for all articulations.
    void ApplyPassiveForces();

private:
    JPH::PhysicsSystem m_physics_system;
    std::unique_ptr<JPH::TempAllocatorImpl> m_temp_allocator;
    std::unique_ptr<JPH::JobSystem> m_local_job_system;
    BPLayerInterfaceImpl m_bp_layer_interface;
    ObjectVsBroadPhaseLayerFilterImpl m_obj_vs_bp_filter;
    ObjectLayerPairFilterImpl m_obj_layer_pair_filter;
    ContactListenerImpl m_contact_listener;
    BodyRegistry m_registry;
    StateSnapshot m_initial_snapshot;
    std::vector<std::unique_ptr<Articulation>> m_articulations;
    bool m_initialized = false;
};

} // namespace joltgym
