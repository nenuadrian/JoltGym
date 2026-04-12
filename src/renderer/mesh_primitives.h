#pragma once

#include "vulkan_context.h"
#include "renderer.h"
#include <vector>

namespace joltgym {

struct Vertex {
    float pos[3];
    float normal[3];
};

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

class MeshPrimitives {
public:
    void Init(VulkanContext& ctx);
    void Shutdown(VulkanContext& ctx);

    // Generate primitive meshes
    static MeshData GenerateCapsule(float half_height, float radius, int segments = 16);
    static MeshData GenerateSphere(float radius, int segments = 16);
    static MeshData GenerateBox(float hx, float hy, float hz);
    static MeshData GenerateCylinder(float half_height, float radius, int segments = 16);
    static MeshData GeneratePlane(float size);

    // GPU buffers for each primitive
    struct GPUMesh {
        VkBuffer vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vertex_alloc = VK_NULL_HANDLE;
        VkBuffer index_buffer = VK_NULL_HANDLE;
        VmaAllocation index_alloc = VK_NULL_HANDLE;
        uint32_t index_count = 0;
    };

    const GPUMesh& GetMesh(PrimitiveType type) const;

private:
    GPUMesh UploadMesh(VulkanContext& ctx, const MeshData& data);
    void DestroyMesh(VulkanContext& ctx, GPUMesh& mesh);

    GPUMesh m_capsule;
    GPUMesh m_sphere;
    GPUMesh m_box;
    GPUMesh m_cylinder;
    GPUMesh m_plane;
};

} // namespace joltgym
