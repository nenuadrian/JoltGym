#pragma once

#include "vulkan_context.h"
#include "mesh_primitives.h"

namespace joltgym {

struct SceneUBO {
    float view[16];
    float proj[16];
    float light_dir[3];
    float ambient;
    float light_color[3];
    float _pad;
    float camera_pos[3];
    float _pad2;
};

struct PushConstants {
    float model[16];
    float color[4];
};

class Pipeline {
public:
    bool Init(VulkanContext& ctx, VkRenderPass render_pass, uint32_t subpass = 0);
    void Shutdown(VulkanContext& ctx);

    void Bind(VkCommandBuffer cmd);
    void PushModel(VkCommandBuffer cmd, const float* model_4x4, const float* color_4);
    void UpdateSceneUBO(VulkanContext& ctx, const SceneUBO& ubo);

    VkDescriptorSet GetDescriptorSet() const { return m_descriptor_set; }
    VkPipelineLayout GetLayout() const { return m_pipeline_layout; }

private:
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptor_set = VK_NULL_HANDLE;
    VkBuffer m_ubo_buffer = VK_NULL_HANDLE;
    VmaAllocation m_ubo_alloc = VK_NULL_HANDLE;
};

} // namespace joltgym
