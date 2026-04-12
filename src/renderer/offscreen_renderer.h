#pragma once

#include "renderer.h"
#include "vulkan_context.h"
#include "pipeline.h"
#include "mesh_primitives.h"
#include <vector>

namespace joltgym {

class OffscreenRenderer : public Renderer {
public:
    ~OffscreenRenderer() override;

    bool Init(uint32_t width, uint32_t height) override;
    void BeginFrame() override;
    void DrawPrimitive(const RenderTransform& transform) override;
    void EndFrame() override;
    void Shutdown() override;

    void SetViewMatrix(const float* view_4x4) override;
    void SetProjectionMatrix(const float* proj_4x4) override;
    void SetLightDirection(float x, float y, float z) override;
    void SetCameraPosition(float x, float y, float z) override;

    std::vector<uint8_t> GetFrameBuffer(uint32_t& width, uint32_t& height) override;

private:
    VulkanContext m_ctx;
    Pipeline m_pipeline;
    MeshPrimitives m_meshes;
    SceneUBO m_scene_ubo;

    VkImage m_color_image = VK_NULL_HANDLE;
    VmaAllocation m_color_alloc = VK_NULL_HANDLE;
    VkImageView m_color_view = VK_NULL_HANDLE;

    VkImage m_depth_image = VK_NULL_HANDLE;
    VmaAllocation m_depth_alloc = VK_NULL_HANDLE;
    VkImageView m_depth_view = VK_NULL_HANDLE;

    VkRenderPass m_render_pass = VK_NULL_HANDLE;
    VkFramebuffer m_framebuffer = VK_NULL_HANDLE;

    VkCommandPool m_cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer m_cmd = VK_NULL_HANDLE;

    // Readback buffer
    VkBuffer m_readback_buffer = VK_NULL_HANDLE;
    VmaAllocation m_readback_alloc = VK_NULL_HANDLE;

    uint32_t m_width = 0, m_height = 0;
};

} // namespace joltgym
