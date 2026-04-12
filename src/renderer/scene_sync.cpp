#include "scene_sync.h"
#include <cmath>
#include <cstring>

namespace joltgym {

void SceneSync::MakeIdentity(float* m) {
    std::memset(m, 0, sizeof(float) * 16);
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

void SceneSync::MakeTranslation(float* m, float x, float y, float z) {
    MakeIdentity(m);
    m[12] = x; m[13] = y; m[14] = z;
}

static PrimitiveType TypeFromString(const std::string& type) {
    if (type == "capsule")  return PrimitiveType::Capsule;
    if (type == "sphere")   return PrimitiveType::Sphere;
    if (type == "box")      return PrimitiveType::Box;
    if (type == "cylinder") return PrimitiveType::Cylinder;
    if (type == "plane")    return PrimitiveType::Plane;
    return PrimitiveType::Capsule;
}

void SceneSync::BuildFromModel(const MjcfModel& model, const BodyRegistry& registry) {
    m_geoms.clear();
    AddBodyGeoms(model.worldbody, registry, true);
}

void SceneSync::AddBodyGeoms(const MjcfBody& body, const BodyRegistry& registry, bool is_world) {
    for (auto& geom : body.geoms) {
        RenderableGeom rg;
        rg.type = TypeFromString(geom.type);
        rg.color[0] = geom.rgba.x;
        rg.color[1] = geom.rgba.y;
        rg.color[2] = geom.rgba.z;
        rg.color[3] = geom.rgba.w;
        rg.is_floor = (geom.type == "plane");

        // Determine scale based on geom type
        if (geom.type == "plane") {
            rg.scale[0] = 50.0f; rg.scale[1] = 1.0f; rg.scale[2] = 50.0f;
            rg.color[3] = 0.1f; // Signal floor for shader
        } else if (geom.fromto.has_value()) {
            auto& ft = *geom.fromto;
            float dx = ft[3]-ft[0], dy = ft[4]-ft[1], dz = ft[5]-ft[2];
            float len = std::sqrt(dx*dx + dy*dy + dz*dz);
            float radius = geom.size.empty() ? 0.046f : geom.size[0];
            rg.scale[0] = radius;
            rg.scale[1] = len * 0.5f; // half-height for capsule
            rg.scale[2] = radius;
        } else if (!geom.size.empty()) {
            rg.scale[0] = geom.size[0];
            rg.scale[1] = geom.size.size() > 1 ? geom.size[1] : geom.size[0];
            rg.scale[2] = geom.size.size() > 2 ? geom.size[2] : geom.size[0];
        } else {
            rg.scale[0] = rg.scale[1] = rg.scale[2] = 0.05f;
        }

        // Body ID for transform lookup
        std::string body_name = is_world ?
            (geom.name.empty() ? "floor" : geom.name) : body.name;
        try {
            rg.body_id = registry.GetBody(body_name);
        } catch (...) {
            continue; // Skip if body not found
        }

        MakeIdentity(rg.local_offset);

        m_geoms.push_back(rg);
    }

    for (auto& child : body.children) {
        AddBodyGeoms(child, registry, false);
    }
}

void SceneSync::Sync(const PhysicsWorld& world, std::vector<RenderTransform>& out) const {
    out.clear();
    auto& body_interface = world.GetPhysicsSystem().GetBodyInterface();

    for (auto& rg : m_geoms) {
        RenderTransform rt;
        rt.type = rg.type;
        std::memcpy(rt.scale, rg.scale, sizeof(float) * 3);
        std::memcpy(rt.color, rg.color, sizeof(float) * 4);

        // Get body world transform
        JPH::RMat44 body_transform = body_interface.GetWorldTransform(rg.body_id);

        // Build model matrix: body_transform * local_offset * scale
        // For now, just use body transform with scale
        float sx = rg.scale[0], sy = rg.scale[1], sz = rg.scale[2];
        auto col0 = body_transform.GetColumn4(0);
        auto col1 = body_transform.GetColumn4(1);
        auto col2 = body_transform.GetColumn4(2);
        auto col3 = body_transform.GetColumn4(3);

        // Column-major 4x4 with scale baked in
        rt.model[0]  = col0.GetX() * sx;
        rt.model[1]  = col0.GetY() * sx;
        rt.model[2]  = col0.GetZ() * sx;
        rt.model[3]  = 0;
        rt.model[4]  = col1.GetX() * sy;
        rt.model[5]  = col1.GetY() * sy;
        rt.model[6]  = col1.GetZ() * sy;
        rt.model[7]  = 0;
        rt.model[8]  = col2.GetX() * sz;
        rt.model[9]  = col2.GetY() * sz;
        rt.model[10] = col2.GetZ() * sz;
        rt.model[11] = 0;
        rt.model[12] = col3.GetX();
        rt.model[13] = col3.GetY();
        rt.model[14] = col3.GetZ();
        rt.model[15] = 1;

        out.push_back(rt);
    }
}

} // namespace joltgym
