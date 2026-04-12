#pragma once

#include "joltgym_core.h"
#include "collision_layers.h"
#include "body_registry.h"

#include <vector>
#include <memory>

namespace joltgym {

class Articulation;

struct StateSnapshot {
    struct BodyState {
        JPH::BodyID id;
        JPH::RVec3 position;
        JPH::Quat rotation;
        JPH::Vec3 linear_velocity;
        JPH::Vec3 angular_velocity;
    };
    std::vector<BodyState> bodies;
};

class PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld();

    // When single_threaded=true, this world uses its own JobSystemSingleThreaded
    // instead of the shared thread pool. Use for batched/pooled worlds.
    void Init(uint32_t max_bodies = 2048, uint32_t max_body_pairs = 4096,
              uint32_t max_contact_constraints = 2048,
              bool single_threaded = false);

    void Step(float dt, int collision_steps = 1);
    void SetGravity(JPH::Vec3 gravity);

    void SaveSnapshot();
    void RestoreSnapshot();

    JPH::PhysicsSystem& GetPhysicsSystem() { return m_physics_system; }
    const JPH::PhysicsSystem& GetPhysicsSystem() const { return m_physics_system; }
    JPH::BodyInterface& GetBodyInterface() { return m_physics_system.GetBodyInterface(); }
    BodyRegistry& GetRegistry() { return m_registry; }
    const BodyRegistry& GetRegistry() const { return m_registry; }

    void AddArticulation(std::unique_ptr<Articulation> articulation);
    const std::vector<std::unique_ptr<Articulation>>& GetArticulations() const { return m_articulations; }
    Articulation* GetArticulation(size_t index) { return m_articulations[index].get(); }

    // Apply damping/stiffness torques for all articulations
    void ApplyPassiveForces();

private:
    JPH::PhysicsSystem m_physics_system;
    std::unique_ptr<JPH::TempAllocatorImpl> m_temp_allocator;
    std::unique_ptr<JPH::JobSystem> m_local_job_system; // null = use shared pool
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
