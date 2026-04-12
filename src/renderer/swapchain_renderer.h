#pragma once

#include "renderer.h"
#include "vulkan_context.h"
#include "pipeline.h"
#include "mesh_primitives.h"
#include "camera.h"

#include <SDL.h>
#include <SDL_vulkan.h>
#include <VkBootstrap.h>
#include <vector>

namespace joltgym {

class ImGuiLayer;

class SwapchainRenderer : public Renderer {
public:
    SwapchainRenderer();
    ~SwapchainRenderer() override;

    bool Init(uint32_t width, uint32_t height) override;
    void BeginFrame() override;
    void DrawPrimitive(const RenderTransform& transform) override;
    void EndFrame() override;
    void Shutdown() override;

    bool ShouldClose() const override { return m_should_close; }
    void PollEvents() override;

    void SetViewMatrix(const float* view_4x4) override;
    void SetProjectionMatrix(const float* proj_4x4) override;
    void SetLightDirection(float x, float y, float z) override;
    void SetCameraPosition(float x, float y, float z) override;

    VulkanContext& GetContext() { return m_ctx; }
    VkRenderPass GetRenderPass() const { return m_render_pass; }

    void SetImGuiLayer(ImGuiLayer* layer) { m_imgui_layer = layer; }

private:
    bool CreateSwapchain();
    bool CreateRenderPass();
    bool CreateFramebuffers();
    bool CreateSyncObjects();
    void CleanupSwapchain();

    SDL_Window* m_window = nullptr;
    VulkanContext m_ctx;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;

    vkb::Swapchain m_vkb_swapchain;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> m_swapchain_images;
    std::vector<VkImageView> m_swapchain_views;
    VkFormat m_swapchain_format;
    VkExtent2D m_swapchain_extent;

    VkRenderPass m_render_pass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> m_framebuffers;

    // Depth buffer
    VkImage m_depth_image = VK_NULL_HANDLE;
    VmaAllocation m_depth_alloc = VK_NULL_HANDLE;
    VkImageView m_depth_view = VK_NULL_HANDLE;

    VkCommandPool m_cmd_pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_cmd_buffers;

    static const int MAX_FRAMES_IN_FLIGHT = 2;
    VkSemaphore m_image_available[MAX_FRAMES_IN_FLIGHT];
    VkSemaphore m_render_finished[MAX_FRAMES_IN_FLIGHT];
    VkFence m_in_flight[MAX_FRAMES_IN_FLIGHT];
    uint32_t m_current_frame = 0;
    uint32_t m_image_index = 0;

    Pipeline m_pipeline;
    MeshPrimitives m_meshes;
    SceneUBO m_scene_ubo;
    ImGuiLayer* m_imgui_layer = nullptr;

    bool m_should_close = false;
    uint32_t m_width, m_height;
};

} // namespace joltgym
