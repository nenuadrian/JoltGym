#include "mesh_primitives.h"
#include <cmath>
#include <stdexcept>

namespace joltgym {

static const float PI = 3.14159265358979323846f;

void MeshPrimitives::Init(VulkanContext& ctx) {
    m_capsule  = UploadMesh(ctx, GenerateCapsule(0.5f, 1.0f));
    m_sphere   = UploadMesh(ctx, GenerateSphere(1.0f));
    m_box      = UploadMesh(ctx, GenerateBox(1.0f, 1.0f, 1.0f));
    m_cylinder = UploadMesh(ctx, GenerateCylinder(0.5f, 1.0f));
    m_plane    = UploadMesh(ctx, GeneratePlane(1.0f));
}

void MeshPrimitives::Shutdown(VulkanContext& ctx) {
    DestroyMesh(ctx, m_capsule);
    DestroyMesh(ctx, m_sphere);
    DestroyMesh(ctx, m_box);
    DestroyMesh(ctx, m_cylinder);
    DestroyMesh(ctx, m_plane);
}

const MeshPrimitives::GPUMesh& MeshPrimitives::GetMesh(PrimitiveType type) const {
    switch (type) {
        case PrimitiveType::Capsule:  return m_capsule;
        case PrimitiveType::Sphere:   return m_sphere;
        case PrimitiveType::Box:      return m_box;
        case PrimitiveType::Cylinder: return m_cylinder;
        case PrimitiveType::Plane:    return m_plane;
        default: return m_box;
    }
}

MeshData MeshPrimitives::GenerateCapsule(float half_height, float radius, int segments) {
    MeshData data;
    int rings = segments / 2;

    // Generate as cylinder + two hemispheres
    // Top hemisphere
    for (int i = 0; i <= rings; i++) {
        float phi = PI * 0.5f * i / rings;
        float y = half_height + radius * std::cos(phi);
        float r = radius * std::sin(phi);

        for (int j = 0; j <= segments; j++) {
            float theta = 2.0f * PI * j / segments;
            Vertex v;
            v.pos[0] = r * std::cos(theta);
            v.pos[1] = y;
            v.pos[2] = r * std::sin(theta);
            float nx = std::sin(phi) * std::cos(theta);
            float ny = std::cos(phi);
            float nz = std::sin(phi) * std::sin(theta);
            v.normal[0] = nx; v.normal[1] = ny; v.normal[2] = nz;
            data.vertices.push_back(v);
        }
    }

    // Cylinder body
    for (int i = 0; i <= 1; i++) {
        float y = half_height - i * 2.0f * half_height;
        for (int j = 0; j <= segments; j++) {
            float theta = 2.0f * PI * j / segments;
            Vertex v;
            v.pos[0] = radius * std::cos(theta);
            v.pos[1] = y;
            v.pos[2] = radius * std::sin(theta);
            v.normal[0] = std::cos(theta);
            v.normal[1] = 0;
            v.normal[2] = std::sin(theta);
            data.vertices.push_back(v);
        }
    }

    // Bottom hemisphere
    for (int i = 0; i <= rings; i++) {
        float phi = PI * 0.5f + PI * 0.5f * i / rings;
        float y = -half_height + radius * std::cos(phi);
        float r = std::abs(radius * std::sin(phi));

        for (int j = 0; j <= segments; j++) {
            float theta = 2.0f * PI * j / segments;
            Vertex v;
            v.pos[0] = r * std::cos(theta);
            v.pos[1] = y;
            v.pos[2] = r * std::sin(theta);
            float nx = std::sin(phi) * std::cos(theta);
            float ny = std::cos(phi);
            float nz = std::sin(phi) * std::sin(theta);
            v.normal[0] = nx; v.normal[1] = ny; v.normal[2] = nz;
            data.vertices.push_back(v);
        }
    }

    // Generate indices
    int total_rings = rings + 2 + rings; // top_hemi + cylinder + bottom_hemi
    int verts_per_ring = segments + 1;
    for (int i = 0; i < total_rings; i++) {
        for (int j = 0; j < segments; j++) {
            uint32_t a = i * verts_per_ring + j;
            uint32_t b = a + verts_per_ring;
            uint32_t c = a + 1;
            uint32_t d = b + 1;
            data.indices.push_back(a);
            data.indices.push_back(b);
            data.indices.push_back(c);
            data.indices.push_back(c);
            data.indices.push_back(b);
            data.indices.push_back(d);
        }
    }

    return data;
}

MeshData MeshPrimitives::GenerateSphere(float radius, int segments) {
    MeshData data;
    int rings = segments;

    for (int i = 0; i <= rings; i++) {
        float phi = PI * i / rings;
        for (int j = 0; j <= segments; j++) {
            float theta = 2.0f * PI * j / segments;
            Vertex v;
            float sp = std::sin(phi), cp = std::cos(phi);
            float st = std::sin(theta), ct = std::cos(theta);
            v.pos[0] = radius * sp * ct;
            v.pos[1] = radius * cp;
            v.pos[2] = radius * sp * st;
            v.normal[0] = sp * ct;
            v.normal[1] = cp;
            v.normal[2] = sp * st;
            data.vertices.push_back(v);
        }
    }

    int verts_per_ring = segments + 1;
    for (int i = 0; i < rings; i++) {
        for (int j = 0; j < segments; j++) {
            uint32_t a = i * verts_per_ring + j;
            uint32_t b = a + verts_per_ring;
            data.indices.push_back(a);
            data.indices.push_back(b);
            data.indices.push_back(a + 1);
            data.indices.push_back(a + 1);
            data.indices.push_back(b);
            data.indices.push_back(b + 1);
        }
    }

    return data;
}

MeshData MeshPrimitives::GenerateBox(float hx, float hy, float hz) {
    MeshData data;
    // 6 faces, 4 vertices each
    struct Face { float nx, ny, nz; float verts[4][3]; };
    Face faces[] = {
        { 0, 0, 1,  {{-hx,-hy,hz},{hx,-hy,hz},{hx,hy,hz},{-hx,hy,hz}} },
        { 0, 0,-1,  {{hx,-hy,-hz},{-hx,-hy,-hz},{-hx,hy,-hz},{hx,hy,-hz}} },
        { 0, 1, 0,  {{-hx,hy,hz},{hx,hy,hz},{hx,hy,-hz},{-hx,hy,-hz}} },
        { 0,-1, 0,  {{-hx,-hy,-hz},{hx,-hy,-hz},{hx,-hy,hz},{-hx,-hy,hz}} },
        { 1, 0, 0,  {{hx,-hy,hz},{hx,-hy,-hz},{hx,hy,-hz},{hx,hy,hz}} },
        {-1, 0, 0,  {{-hx,-hy,-hz},{-hx,-hy,hz},{-hx,hy,hz},{-hx,hy,-hz}} },
    };

    for (auto& f : faces) {
        uint32_t base = (uint32_t)data.vertices.size();
        for (int i = 0; i < 4; i++) {
            Vertex v;
            v.pos[0] = f.verts[i][0]; v.pos[1] = f.verts[i][1]; v.pos[2] = f.verts[i][2];
            v.normal[0] = f.nx; v.normal[1] = f.ny; v.normal[2] = f.nz;
            data.vertices.push_back(v);
        }
        data.indices.push_back(base);     data.indices.push_back(base + 1); data.indices.push_back(base + 2);
        data.indices.push_back(base);     data.indices.push_back(base + 2); data.indices.push_back(base + 3);
    }

    return data;
}

MeshData MeshPrimitives::GenerateCylinder(float half_height, float radius, int segments) {
    MeshData data;

    // Barrel
    for (int i = 0; i <= 1; i++) {
        float y = half_height - i * 2.0f * half_height;
        for (int j = 0; j <= segments; j++) {
            float theta = 2.0f * PI * j / segments;
            Vertex v;
            v.pos[0] = radius * std::cos(theta);
            v.pos[1] = y;
            v.pos[2] = radius * std::sin(theta);
            v.normal[0] = std::cos(theta);
            v.normal[1] = 0;
            v.normal[2] = std::sin(theta);
            data.vertices.push_back(v);
        }
    }

    int vpr = segments + 1;
    for (int j = 0; j < segments; j++) {
        data.indices.push_back(j);
        data.indices.push_back(j + vpr);
        data.indices.push_back(j + 1);
        data.indices.push_back(j + 1);
        data.indices.push_back(j + vpr);
        data.indices.push_back(j + vpr + 1);
    }

    // Top cap
    uint32_t center_top = (uint32_t)data.vertices.size();
    data.vertices.push_back({{0, half_height, 0}, {0, 1, 0}});
    for (int j = 0; j <= segments; j++) {
        float theta = 2.0f * PI * j / segments;
        data.vertices.push_back({{radius * std::cos(theta), half_height, radius * std::sin(theta)}, {0, 1, 0}});
    }
    for (int j = 0; j < segments; j++) {
        data.indices.push_back(center_top);
        data.indices.push_back(center_top + 1 + j);
        data.indices.push_back(center_top + 2 + j);
    }

    // Bottom cap
    uint32_t center_bot = (uint32_t)data.vertices.size();
    data.vertices.push_back({{0, -half_height, 0}, {0, -1, 0}});
    for (int j = 0; j <= segments; j++) {
        float theta = 2.0f * PI * j / segments;
        data.vertices.push_back({{radius * std::cos(theta), -half_height, radius * std::sin(theta)}, {0, -1, 0}});
    }
    for (int j = 0; j < segments; j++) {
        data.indices.push_back(center_bot);
        data.indices.push_back(center_bot + 2 + j);
        data.indices.push_back(center_bot + 1 + j);
    }

    return data;
}

MeshData MeshPrimitives::GeneratePlane(float size) {
    MeshData data;
    data.vertices = {
        {{-size, 0, -size}, {0, 1, 0}},
        {{ size, 0, -size}, {0, 1, 0}},
        {{ size, 0,  size}, {0, 1, 0}},
        {{-size, 0,  size}, {0, 1, 0}},
    };
    data.indices = {0, 1, 2, 0, 2, 3};
    return data;
}

MeshPrimitives::GPUMesh MeshPrimitives::UploadMesh(VulkanContext& ctx, const MeshData& data) {
    GPUMesh mesh;
    mesh.index_count = (uint32_t)data.indices.size();

    // Vertex buffer
    VkBufferCreateInfo vb_info = {};
    vb_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vb_info.size = data.vertices.size() * sizeof(Vertex);
    vb_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    VmaAllocationCreateInfo vma_info = {};
    vma_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    vmaCreateBuffer(ctx.GetAllocator(), &vb_info, &vma_info,
                    &mesh.vertex_buffer, &mesh.vertex_alloc, nullptr);

    void* mapped;
    vmaMapMemory(ctx.GetAllocator(), mesh.vertex_alloc, &mapped);
    memcpy(mapped, data.vertices.data(), vb_info.size);
    vmaUnmapMemory(ctx.GetAllocator(), mesh.vertex_alloc);

    // Index buffer
    VkBufferCreateInfo ib_info = {};
    ib_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ib_info.size = data.indices.size() * sizeof(uint32_t);
    ib_info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    vmaCreateBuffer(ctx.GetAllocator(), &ib_info, &vma_info,
                    &mesh.index_buffer, &mesh.index_alloc, nullptr);

    vmaMapMemory(ctx.GetAllocator(), mesh.index_alloc, &mapped);
    memcpy(mapped, data.indices.data(), ib_info.size);
    vmaUnmapMemory(ctx.GetAllocator(), mesh.index_alloc);

    return mesh;
}

void MeshPrimitives::DestroyMesh(VulkanContext& ctx, GPUMesh& mesh) {
    if (mesh.vertex_buffer) vmaDestroyBuffer(ctx.GetAllocator(), mesh.vertex_buffer, mesh.vertex_alloc);
    if (mesh.index_buffer)  vmaDestroyBuffer(ctx.GetAllocator(), mesh.index_buffer, mesh.index_alloc);
    mesh = {};
}

} // namespace joltgym
