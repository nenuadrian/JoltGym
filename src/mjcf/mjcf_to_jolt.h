#pragma once

#include "mjcf_model.h"
#include "core/physics_world.h"
#include "core/articulation.h"
#include <Jolt/Jolt.h>

namespace joltgym {

class MjcfToJolt {
public:
    // Build a PhysicsWorld from an MjcfModel
    // Returns the articulation created for the main body
    Articulation* Build(const MjcfModel& model, PhysicsWorld& world,
                        JPH::Vec3 offset = JPH::Vec3::sZero(),
                        bool create_floor = true);

private:
    // Recursive body builder
    void BuildBody(const MjcfBody& mjcf_body,
                   JPH::BodyID parent_body_id,
                   JPH::RVec3 parent_world_pos,
                   JPH::Quat parent_world_rot,
                   PhysicsWorld& world,
                   Articulation& articulation,
                   bool is_root);

    // Shape creation
    JPH::ShapeRefC CreateShape(const MjcfGeom& geom);
    JPH::ShapeRefC CreateCapsuleShape(const MjcfGeom& geom);
    JPH::ShapeRefC CreateBoxShape(const MjcfGeom& geom);
    JPH::ShapeRefC CreateSphereShape(const MjcfGeom& geom);
    JPH::ShapeRefC CreatePlaneShape(const MjcfGeom& geom);

    // Create compound shape from multiple geoms
    JPH::ShapeRefC CreateCompoundShape(const std::vector<MjcfGeom>& geoms);

    // Constraint creation
    void CreateHingeConstraint(const MjcfJoint& joint,
                               JPH::BodyID parent_id, JPH::BodyID child_id,
                               JPH::RVec3 world_pos, JPH::Quat world_rot,
                               PhysicsWorld& world, Articulation& articulation);

    // Mass scaling
    void ApplyTotalMassScaling(PhysicsWorld& world, float target_mass);

    // Helpers
    JPH::Quat AxisAngleToQuat(float ax, float ay, float az, float angle);
    JPH::Vec3 ToJoltVec3(const Vec3f& v);
    JPH::Quat ComputeRotationFromTo(JPH::Vec3 from, JPH::Vec3 to);

    const MjcfModel* m_model = nullptr;
    JPH::Vec3 m_offset = JPH::Vec3::sZero();

    // Track all dynamic body IDs for mass scaling
    std::vector<JPH::BodyID> m_dynamic_bodies;
};

} // namespace joltgym
