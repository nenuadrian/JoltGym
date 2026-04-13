#include "mjcf_to_jolt.h"
#include "core/motor_controller.h"
#include <cmath>
#include <stdexcept>

namespace joltgym {

JPH::Vec3 MjcfToJolt::ToJoltVec3(const Vec3f& v) {
    return JPH::Vec3(v.x, v.y, v.z);
}

JPH::Quat MjcfToJolt::AxisAngleToQuat(float ax, float ay, float az, float angle) {
    float half = angle * 0.5f;
    float s = std::sin(half);
    float len = std::sqrt(ax*ax + ay*ay + az*az);
    if (len < 1e-8f) return JPH::Quat::sIdentity();
    return JPH::Quat(ax/len * s, ay/len * s, az/len * s, std::cos(half));
}

JPH::Quat MjcfToJolt::ComputeRotationFromTo(JPH::Vec3 from, JPH::Vec3 to) {
    from = from.Normalized();
    to = to.Normalized();
    float d = from.Dot(to);
    if (d > 0.9999f) return JPH::Quat::sIdentity();
    if (d < -0.9999f) {
        // 180 degree rotation - pick an arbitrary perpendicular axis
        JPH::Vec3 perp = from.GetNormalizedPerpendicular();
        return JPH::Quat(perp.GetX(), perp.GetY(), perp.GetZ(), 0.0f);
    }
    JPH::Vec3 cross = from.Cross(to);
    return JPH::Quat(cross.GetX(), cross.GetY(), cross.GetZ(), 1.0f + d).Normalized();
}

Articulation* MjcfToJolt::Build(const MjcfModel& model, PhysicsWorld& world,
                                 JPH::Vec3 offset, bool create_floor) {
    m_model = &model;
    m_offset = offset;
    m_dynamic_bodies.clear();

    // Set gravity from model
    world.SetGravity(JPH::Vec3(model.option.gravity.x,
                                model.option.gravity.y,
                                model.option.gravity.z));

    // Create floor from worldbody geoms (only once for multi-agent)
    auto& body_interface = world.GetBodyInterface();
    for (auto& geom : model.worldbody.geoms) {
        if (!create_floor) break;
        if (geom.type == "plane") {
            JPH::BoxShapeSettings floor_shape(JPH::Vec3(100.0f, 100.0f, 0.5f));
            floor_shape.SetEmbedded();
            auto shape_result = floor_shape.Create();
            if (shape_result.HasError()) {
                fprintf(stderr, "Warning: floor shape creation failed\n");
                continue;
            }
            JPH::BodyCreationSettings floor_settings(
                shape_result.Get(),
                JPH::RVec3(0, 0, -0.5) + JPH::RVec3(m_offset),
                JPH::Quat::sIdentity(),
                JPH::EMotionType::Static,
                joltgym::Layers::STATIC);
            floor_settings.mFriction = geom.friction;
            auto floor_id = body_interface.CreateAndAddBody(
                floor_settings, JPH::EActivation::DontActivate);
            world.GetRegistry().RegisterBody(
                geom.name.empty() ? "floor" : geom.name, floor_id);
        }
    }

    // Build each root body in worldbody
    for (auto& child : model.worldbody.children) {
        auto articulation = std::make_unique<Articulation>(child.name);

        BuildBody(child, JPH::BodyID(), JPH::RVec3(m_offset),
                  JPH::Quat::sIdentity(), world, *articulation, true);

        // Create actuators/motors
        for (auto& act : model.actuators) {
            auto* constraint = world.GetRegistry().GetConstraint(act.joint);
            if (!constraint) continue;

            // Find the joint to get damping/stiffness/armature
            float damping = 0, stiffness = 0, armature = 0;
            std::function<bool(const MjcfBody&)> findJoint = [&](const MjcfBody& body) -> bool {
                for (auto& j : body.joints) {
                    if (j.name == act.joint) {
                        damping = j.damping;
                        stiffness = j.stiffness;
                        armature = j.armature;
                        return true;
                    }
                }
                for (auto& c : body.children) {
                    if (findJoint(c)) return true;
                }
                return false;
            };
            findJoint(model.worldbody);

            auto motor = std::make_unique<MotorController>(
                act.name, constraint, MotorController::JointType::Hinge,
                act.gear, act.ctrl_min, act.ctrl_max,
                damping, stiffness, armature);
            articulation->AddMotor(std::move(motor));
        }

        // Apply total mass scaling if specified
        if (model.compiler.settotalmass > 0) {
            ApplyTotalMassScaling(world, model.compiler.settotalmass);
        }

        Articulation* ptr = articulation.get();
        world.AddArticulation(std::move(articulation));

        // Save initial state for reset
        world.SaveSnapshot();

        return ptr;
    }

    return nullptr;
}

JPH::BodyID MjcfToJolt::CreateIntermediateBody(JPH::RVec3 world_pos, JPH::Quat world_rot,
                                                  PhysicsWorld& world, Articulation& articulation,
                                                  const std::string& name) {
    auto& body_interface = world.GetBodyInterface();

    // Tiny sphere — kinematic connector for multi-joint decomposition
    auto shape = JPH::ShapeRefC(new JPH::SphereShape(0.005f));

    JPH::BodyCreationSettings settings(
        shape, world_pos, world_rot,
        JPH::EMotionType::Dynamic, Layers::DYNAMIC);
    settings.mAllowSleeping = false;
    settings.mLinearDamping = 0.0f;
    settings.mAngularDamping = 0.5f; // Mild damping to prevent oscillation
    settings.mFriction = 0.0f;
    settings.mRestitution = 0.0f;

    // Small mass — enough for stable constraint solving, not enough to affect dynamics
    settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    settings.mMassPropertiesOverride.mMass = 0.05f;

    auto body_id = body_interface.CreateAndAddBody(settings, JPH::EActivation::Activate);
    world.GetRegistry().RegisterBody(name, body_id);
    articulation.AddBody(name, body_id);
    m_dynamic_bodies.push_back(body_id);

    return body_id;
}

void MjcfToJolt::BuildBody(const MjcfBody& mjcf_body,
                            JPH::BodyID parent_body_id,
                            JPH::RVec3 parent_world_pos,
                            JPH::Quat parent_world_rot,
                            PhysicsWorld& world,
                            Articulation& articulation,
                            bool is_root) {
    auto& body_interface = world.GetBodyInterface();

    // Compute world position for this body
    JPH::Vec3 local_pos = ToJoltVec3(mjcf_body.pos);
    JPH::RVec3 world_pos = parent_world_pos + JPH::RVec3(parent_world_rot * local_pos);

    // Apply body quaternion if specified
    JPH::Quat world_rot = parent_world_rot;
    if (mjcf_body.has_quat) {
        JPH::Quat body_quat(mjcf_body.quat.x, mjcf_body.quat.y,
                             mjcf_body.quat.z, mjcf_body.quat.w);
        world_rot = parent_world_rot * body_quat;
    }

    // Create the shape from geoms
    JPH::ShapeRefC shape;
    if (mjcf_body.geoms.size() == 1) {
        shape = CreateShape(mjcf_body.geoms[0]);
    } else if (mjcf_body.geoms.size() > 1) {
        shape = CreateCompoundShape(mjcf_body.geoms);
    } else {
        // Body with no geoms — create a tiny sphere as placeholder
        shape = new JPH::SphereShape(0.01f);
    }

    if (!shape) {
        throw std::runtime_error("Failed to create shape for body: " + mjcf_body.name);
    }

    // Create body
    JPH::BodyCreationSettings body_settings(
        shape, world_pos, world_rot,
        JPH::EMotionType::Dynamic, Layers::DYNAMIC);
    body_settings.mAllowSleeping = false; // RL bodies should never sleep
    body_settings.mLinearDamping = 0.0f;
    body_settings.mAngularDamping = 0.0f; // We handle damping ourselves

    // Set friction from MJCF geom (Jolt defaults to 0.2 which is too low)
    if (!mjcf_body.geoms.empty()) {
        body_settings.mFriction = mjcf_body.geoms[0].friction;
    } else {
        body_settings.mFriction = 0.8f; // Sensible default for RL bodies
    }
    body_settings.mRestitution = 0.0f; // No bounce for stable contacts

    auto body_id = body_interface.CreateAndAddBody(body_settings, JPH::EActivation::Activate);
    world.GetRegistry().RegisterBody(mjcf_body.name, body_id);
    articulation.AddBody(mjcf_body.name, body_id);
    m_dynamic_bodies.push_back(body_id);

    if (is_root) {
        articulation.SetRootBody(body_id);

        // Check for free joint first
        bool has_free_joint = false;
        for (auto& joint : mjcf_body.joints) {
            if (joint.type == "free") {
                has_free_joint = true;
                break;
            }
        }

        if (has_free_joint) {
            // Free joint: 7 qpos DOFs (x, y, z, qw, qx, qy, qz)
            articulation.SetHasFreeRoot(true);
            articulation.AddRootDOF({"free_x", RootDOF::Type::FreeX});
            articulation.AddRootDOF({"free_y", RootDOF::Type::FreeY});
            articulation.AddRootDOF({"free_z", RootDOF::Type::FreeZ});
            articulation.AddRootDOF({"free_qw", RootDOF::Type::QuatW});
            articulation.AddRootDOF({"free_qx", RootDOF::Type::QuatX});
            articulation.AddRootDOF({"free_qy", RootDOF::Type::QuatY});
            articulation.AddRootDOF({"free_qz", RootDOF::Type::QuatZ});
        } else {
            // Register individual slide/hinge root DOFs
            for (auto& joint : mjcf_body.joints) {
                if (joint.type == "slide") {
                    if (std::abs(joint.axis.x) > 0.5f) {
                        articulation.AddRootDOF({joint.name, RootDOF::Type::SlideX});
                    } else if (std::abs(joint.axis.z) > 0.5f) {
                        articulation.AddRootDOF({joint.name, RootDOF::Type::SlideZ});
                    }
                } else if (joint.type == "hinge") {
                    if (std::abs(joint.axis.y) > 0.5f) {
                        articulation.AddRootDOF({joint.name, RootDOF::Type::HingeY});
                    }
                }
            }
        }
    }

    // Create constraints for non-root joints
    if (!is_root && !parent_body_id.IsInvalid()) {
        auto& joints = mjcf_body.joints;
        int num_joints = (int)joints.size();

        if (num_joints == 1 && joints[0].type == "hinge") {
            // Simple case: single hinge joint
            CreateHingeConstraint(joints[0], parent_body_id, body_id,
                                  world_pos, world_rot, world, articulation);
        } else if (num_joints > 1) {
            // Multi-joint body: create intermediate bodies
            // Chain: parent → joint[0] → inter[0] → joint[1] → inter[1] → ... → joint[N-1] → actual_body
            JPH::BodyID prev_body_id = parent_body_id;
            for (int ji = 0; ji < num_joints; ji++) {
                auto& joint = joints[ji];
                if (joint.type != "hinge") continue;

                JPH::BodyID next_body_id;
                if (ji < num_joints - 1) {
                    // Create intermediate body
                    std::string inter_name = mjcf_body.name + "_inter" + std::to_string(ji);
                    next_body_id = CreateIntermediateBody(world_pos, world_rot,
                                                          world, articulation, inter_name);
                } else {
                    // Last joint connects to the actual body
                    next_body_id = body_id;
                }

                CreateHingeConstraint(joint, prev_body_id, next_body_id,
                                      world_pos, world_rot, world, articulation);
                prev_body_id = next_body_id;
            }
        }
        // Bodies with 0 joints (like feet) have no constraints — they're just
        // attached by being children in the body tree. We need a fixed constraint.
        if (num_joints == 0) {
            JPH::FixedConstraintSettings fixed_settings;
            fixed_settings.mAutoDetectPoint = true;
            auto& no_lock = world.GetPhysicsSystem().GetBodyLockInterfaceNoLock();
            JPH::Body* parent_body = no_lock.TryGetBody(parent_body_id);
            JPH::Body* child_body = no_lock.TryGetBody(body_id);
            if (parent_body && child_body) {
                auto* constraint = fixed_settings.Create(*parent_body, *child_body);
                world.GetPhysicsSystem().AddConstraint(constraint);
            }
        }
    }

    // Recurse into children
    for (auto& child : mjcf_body.children) {
        BuildBody(child, body_id, world_pos, world_rot, world, articulation, false);
    }
}

void MjcfToJolt::CreateHingeConstraint(const MjcfJoint& joint,
                                         JPH::BodyID parent_id, JPH::BodyID child_id,
                                         JPH::RVec3 world_pos, JPH::Quat world_rot,
                                         PhysicsWorld& world, Articulation& articulation) {
    auto& body_interface = world.GetBodyInterface();

    // Joint position in world space
    JPH::RVec3 joint_world_pos = world_pos + JPH::RVec3(world_rot * ToJoltVec3(joint.pos));

    // Joint axis in world space — normalize arbitrary axes like "2 1 1"
    JPH::Vec3 raw_axis = ToJoltVec3(joint.axis);
    float axis_len = raw_axis.Length();
    JPH::Vec3 hinge_axis = (axis_len > 1e-6f) ? (world_rot * raw_axis).Normalized()
                                                : JPH::Vec3(0, 1, 0);

    // Compute a perpendicular normal axis
    JPH::Vec3 normal_axis = hinge_axis.GetNormalizedPerpendicular();

    JPH::HingeConstraintSettings settings;
    settings.mPoint1 = joint_world_pos;
    settings.mPoint2 = joint_world_pos;
    settings.mHingeAxis1 = hinge_axis;
    settings.mHingeAxis2 = hinge_axis;
    settings.mNormalAxis1 = normal_axis;
    settings.mNormalAxis2 = normal_axis;

    if (joint.limited) {
        settings.mLimitsMin = joint.range_min;
        settings.mLimitsMax = joint.range_max;
    } else {
        settings.mLimitsMin = -JPH::JPH_PI;
        settings.mLimitsMax = JPH::JPH_PI;
    }

    // Configure motor settings
    settings.mMotorSettings.mMaxTorqueLimit = 1000.0f;
    settings.mMotorSettings.mMinTorqueLimit = -1000.0f;

    // Use the no-lock interface during construction (single-threaded, no Update running)
    auto& no_lock = world.GetPhysicsSystem().GetBodyLockInterfaceNoLock();
    JPH::Body* parent_body = no_lock.TryGetBody(parent_id);
    JPH::Body* child_body = no_lock.TryGetBody(child_id);

    if (!parent_body || !child_body) return;

    auto* constraint = settings.Create(*parent_body, *child_body);

    world.GetPhysicsSystem().AddConstraint(constraint);
    world.GetRegistry().RegisterConstraint(joint.name, constraint);
}

JPH::ShapeRefC MjcfToJolt::CreateShape(const MjcfGeom& geom) {
    if (geom.type == "capsule") return CreateCapsuleShape(geom);
    if (geom.type == "box")     return CreateBoxShape(geom);
    if (geom.type == "sphere")  return CreateSphereShape(geom);
    if (geom.type == "plane")   return CreatePlaneShape(geom);
    // Default to capsule
    return CreateCapsuleShape(geom);
}

JPH::ShapeRefC MjcfToJolt::CreateCapsuleShape(const MjcfGeom& geom) {
    float radius = 0.05f;
    if (!geom.size.empty()) radius = geom.size[0];

    if (geom.fromto.has_value()) {
        // fromto defines two endpoints
        auto& ft = *geom.fromto;
        JPH::Vec3 p1(ft[0], ft[1], ft[2]);
        JPH::Vec3 p2(ft[3], ft[4], ft[5]);
        JPH::Vec3 dir = p2 - p1;
        float length = dir.Length();
        float half_height = length * 0.5f;
        JPH::Vec3 center = (p1 + p2) * 0.5f;

        if (half_height < 1e-6f) {
            return new JPH::SphereShape(radius);
        }

        auto capsule = new JPH::CapsuleShape(half_height, radius);

        // Capsule in Jolt is along Y axis — rotate to align with direction
        JPH::Vec3 capsule_default_axis(0, 1, 0);
        JPH::Quat rotation = ComputeRotationFromTo(capsule_default_axis, dir.Normalized());

        JPH::RotatedTranslatedShapeSettings rts(
            JPH::Vec3(center.GetX(), center.GetY(), center.GetZ()),
            rotation, capsule);
        auto result = rts.Create();
        if (result.HasError()) return new JPH::SphereShape(radius);
        return result.Get();
    }

    if (geom.size.size() >= 2) {
        // size = [radius, half_height]
        float half_height = geom.size[1];
        auto capsule = new JPH::CapsuleShape(half_height, radius);

        // Apply axisangle rotation and position offset
        JPH::Quat rot = AxisAngleToQuat(geom.axisangle[0], geom.axisangle[1],
                                          geom.axisangle[2], geom.axisangle[3]);
        if (geom.pos.x != 0 || geom.pos.y != 0 || geom.pos.z != 0 ||
            geom.axisangle[3] != 0) {
            JPH::RotatedTranslatedShapeSettings rts(
                ToJoltVec3(geom.pos), rot, capsule);
            auto result = rts.Create();
            if (!result.HasError()) return result.Get();
        }
        return capsule;
    }

    // Minimal capsule
    return new JPH::CapsuleShape(0.1f, radius);
}

JPH::ShapeRefC MjcfToJolt::CreateBoxShape(const MjcfGeom& geom) {
    JPH::Vec3 half_extents(0.1f, 0.1f, 0.1f);
    if (geom.size.size() >= 3) {
        half_extents = JPH::Vec3(geom.size[0], geom.size[1], geom.size[2]);
    }
    return new JPH::BoxShape(half_extents);
}

JPH::ShapeRefC MjcfToJolt::CreateSphereShape(const MjcfGeom& geom) {
    float radius = 0.05f;
    if (!geom.size.empty()) radius = geom.size[0];

    // Apply position offset if specified
    if (geom.pos.x != 0 || geom.pos.y != 0 || geom.pos.z != 0) {
        auto sphere = new JPH::SphereShape(radius);
        JPH::RotatedTranslatedShapeSettings rts(
            ToJoltVec3(geom.pos), JPH::Quat::sIdentity(), sphere);
        auto result = rts.Create();
        if (!result.HasError()) return result.Get();
        return sphere;
    }

    return new JPH::SphereShape(radius);
}

JPH::ShapeRefC MjcfToJolt::CreatePlaneShape(const MjcfGeom& geom) {
    return new JPH::BoxShape(JPH::Vec3(100.0f, 100.0f, 0.5f));
}

JPH::ShapeRefC MjcfToJolt::CreateCompoundShape(const std::vector<MjcfGeom>& geoms) {
    JPH::StaticCompoundShapeSettings compound;
    for (auto& geom : geoms) {
        auto shape = CreateShape(geom);
        if (shape) {
            // For fromto geoms, the position offset is already baked into the RotatedTranslatedShape
            // For pos-based geoms, we might need to add an offset
            compound.AddShape(JPH::Vec3::sZero(), JPH::Quat::sIdentity(), shape);
        }
    }
    auto result = compound.Create();
    if (result.HasError()) return nullptr;
    return result.Get();
}

void MjcfToJolt::ApplyTotalMassScaling(PhysicsWorld& world, float target_mass) {
    auto& body_interface = world.GetBodyInterface();

    auto& no_lock = world.GetPhysicsSystem().GetBodyLockInterfaceNoLock();

    // Sum current masses
    float total_mass = 0;
    for (auto& body_id : m_dynamic_bodies) {
        JPH::Body* body = no_lock.TryGetBody(body_id);
        if (body && body->IsDynamic()) {
            total_mass += 1.0f / body->GetMotionProperties()->GetInverseMass();
        }
    }

    if (total_mass <= 0) return;
    float scale = target_mass / total_mass;

    // Scale each body's mass
    for (auto& body_id : m_dynamic_bodies) {
        JPH::Body* body = no_lock.TryGetBody(body_id);
        if (body && body->IsDynamic()) {
            auto* mp = body->GetMotionProperties();
            float old_mass = 1.0f / mp->GetInverseMass();
            float new_mass = old_mass * scale;
            mp->SetInverseMass(1.0f / new_mass);
            // Scale inverse inertia diagonal proportionally
            JPH::Vec3 inv_diag = mp->GetInverseInertiaDiagonal();
            mp->SetInverseInertia(
                inv_diag * (1.0f / scale),
                mp->GetInertiaRotation());
        }
    }
}

} // namespace joltgym
