#include "physics_world.h"
#include "articulation.h"
#include <Jolt/Core/JobSystemSingleThreaded.h>

namespace joltgym {

PhysicsWorld::PhysicsWorld() {
}

PhysicsWorld::~PhysicsWorld() {
    if (m_initialized) {
        // Remove all bodies
        auto& body_interface = m_physics_system.GetBodyInterface();
        for (auto& body_id : m_registry.GetOrderedBodies()) {
            if (!body_id.IsInvalid()) {
                body_interface.RemoveBody(body_id);
                body_interface.DestroyBody(body_id);
            }
        }
    }
}

void PhysicsWorld::Init(uint32_t max_bodies, uint32_t max_body_pairs,
                        uint32_t max_contact_constraints, bool single_threaded) {
    JoltGymCore::Init();

    m_temp_allocator = std::make_unique<JPH::TempAllocatorImpl>(10 * 1024 * 1024);

    if (single_threaded) {
        m_local_job_system = std::make_unique<JPH::JobSystemSingleThreaded>(JPH::cMaxPhysicsJobs);
    }

    m_physics_system.Init(
        max_bodies, 0, max_body_pairs, max_contact_constraints,
        m_bp_layer_interface, m_obj_vs_bp_filter, m_obj_layer_pair_filter);

    m_physics_system.SetContactListener(&m_contact_listener);
    m_physics_system.SetGravity(JPH::Vec3(0, 0, -9.81f)); // Z-up like MuJoCo

    m_initialized = true;
}

void PhysicsWorld::Step(float dt, int collision_steps) {
    ApplyPassiveForces();
    JPH::JobSystem& js = m_local_job_system
        ? *m_local_job_system
        : static_cast<JPH::JobSystem&>(JoltGymCore::GetJobSystem());
    m_physics_system.Update(dt, collision_steps, m_temp_allocator.get(), &js);
}

void PhysicsWorld::SetGravity(JPH::Vec3 gravity) {
    m_physics_system.SetGravity(gravity);
}

void PhysicsWorld::SaveSnapshot() {
    auto& body_interface = m_physics_system.GetBodyInterface();
    m_initial_snapshot.bodies.clear();

    for (auto& body_id : m_registry.GetOrderedBodies()) {
        StateSnapshot::BodyState state;
        state.id = body_id;
        state.position = body_interface.GetPosition(body_id);
        state.rotation = body_interface.GetRotation(body_id);
        state.linear_velocity = body_interface.GetLinearVelocity(body_id);
        state.angular_velocity = body_interface.GetAngularVelocity(body_id);
        m_initial_snapshot.bodies.push_back(state);
    }
}

void PhysicsWorld::RestoreSnapshot() {
    auto& body_interface = m_physics_system.GetBodyInterface();

    for (auto& state : m_initial_snapshot.bodies) {
        body_interface.SetPositionAndRotation(
            state.id, state.position, state.rotation, JPH::EActivation::Activate);
        body_interface.SetLinearVelocity(state.id, state.linear_velocity);
        body_interface.SetAngularVelocity(state.id, state.angular_velocity);
    }
}

void PhysicsWorld::AddArticulation(std::unique_ptr<Articulation> articulation) {
    m_articulations.push_back(std::move(articulation));
}

void PhysicsWorld::ApplyPassiveForces() {
    for (auto& artic : m_articulations) {
        artic->ApplyPassiveForces(m_physics_system.GetBodyInterface());
    }
}

} // namespace joltgym
