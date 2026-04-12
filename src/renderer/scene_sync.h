#pragma once

#include "renderer.h"
#include "core/physics_world.h"
#include "mjcf/mjcf_model.h"
#include <vector>

namespace joltgym {

struct RenderableGeom {
    PrimitiveType type;
    float scale[3];
    float color[4];
    JPH::BodyID body_id;
    float local_offset[16]; // 4x4 local transform within body
    bool is_floor;
};

class SceneSync {
public:
    void BuildFromModel(const MjcfModel& model, const BodyRegistry& registry);

    // Populate render transforms from current physics state
    void Sync(const PhysicsWorld& world, std::vector<RenderTransform>& out) const;

    const std::vector<RenderableGeom>& GetGeoms() const { return m_geoms; }

private:
    void AddBodyGeoms(const MjcfBody& body, const BodyRegistry& registry, bool is_world);
    void MakeIdentity(float* m);
    void MakeTranslation(float* m, float x, float y, float z);

    std::vector<RenderableGeom> m_geoms;
};

} // namespace joltgym
