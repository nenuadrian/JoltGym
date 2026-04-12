#include "pipeline.h"
#include "phong_vert.h"
#include "phong_frag.h"
#include <cstring>
#include <stdexcept>

namespace joltgym {

bool Pipeline::Init(VulkanContext& ctx, VkRenderPass render_pass, uint32_t subpass) {
    VkDevice device = ctx.GetDevice();

    // Create UBO buffer
    VkBufferCreateInfo ubo_info = {};
    ubo_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ubo_info.size = sizeof(SceneUBO);
    ubo_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    VmaAllocationCreateInfo ubo_alloc_info = {};
    ubo_alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    vmaCreateBuffer(ctx.GetAllocator(), &ubo_info, &ubo_alloc_info,
                    &m_ubo_buffer, &m_ubo_alloc, nullptr);

    // Descriptor set layout
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 1;
    layout_info.pBindings = &binding;
    vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &m_descriptor_set_layout);

    // Descriptor pool
    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_size.descriptorCount = 1;

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    vkCreateDescriptorPool(device, &pool_info, nullptr, &m_descriptor_pool);

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &m_descriptor_set_layout;
    vkAllocateDescriptorSets(device, &alloc_info, &m_descriptor_set);

    // Update descriptor set
    VkDescriptorBufferInfo buf_info = {};
    buf_info.buffer = m_ubo_buffer;
    buf_info.offset = 0;
    buf_info.range = sizeof(SceneUBO);

    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptor_set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &buf_info;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    // Push constant range
    VkPushConstantRange push_range = {};
    push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(PushConstants);

    // Pipeline layout
    VkPipelineLayoutCreateInfo pl_layout_info = {};
    pl_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_layout_info.setLayoutCount = 1;
    pl_layout_info.pSetLayouts = &m_descriptor_set_layout;
    pl_layout_info.pushConstantRangeCount = 1;
    pl_layout_info.pPushConstantRanges = &push_range;
    vkCreatePipelineLayout(device, &pl_layout_info, nullptr, &m_pipeline_layout);

    // Shader modules
    VkShaderModuleCreateInfo vert_info = {};
    vert_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vert_info.codeSize = phong_vert_size;
    vert_info.pCode = reinterpret_cast<const uint32_t*>(phong_vert_data);

    VkShaderModuleCreateInfo frag_info = {};
    frag_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    frag_info.codeSize = phong_frag_size;
    frag_info.pCode = reinterpret_cast<const uint32_t*>(phong_frag_data);

    VkShaderModule vert_module, frag_module;
    vkCreateShaderModule(device, &vert_info, nullptr, &vert_module);
    vkCreateShaderModule(device, &frag_info, nullptr, &frag_module);

    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_module;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_module;
    stages[1].pName = "main";

    // Vertex input
    VkVertexInputBindingDescription vtx_binding = {};
    vtx_binding.binding = 0;
    vtx_binding.stride = sizeof(Vertex);
    vtx_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vtx_attrs[2] = {};
    vtx_attrs[0].location = 0;
    vtx_attrs[0].binding = 0;
    vtx_attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    vtx_attrs[0].offset = offsetof(Vertex, pos);
    vtx_attrs[1].location = 1;
    vtx_attrs[1].binding = 0;
    vtx_attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    vtx_attrs[1].offset = offsetof(Vertex, normal);

    VkPipelineVertexInputStateCreateInfo vertex_input = {};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &vtx_binding;
    vertex_input.vertexAttributeDescriptionCount = 2;
    vertex_input.pVertexAttributeDescriptions = vtx_attrs;

    VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_state = {};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample = {};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth_stencil = {};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = VK_TRUE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blend_attachment = {};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend = {};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;

    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic = {};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;

    VkGraphicsPipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = stages;
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisample;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &blend;
    pipeline_info.pDynamicState = &dynamic;
    pipeline_info.layout = m_pipeline_layout;
    pipeline_info.renderPass = render_pass;
    pipeline_info.subpass = subpass;

    auto result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &m_pipeline);

    vkDestroyShaderModule(device, vert_module, nullptr);
    vkDestroyShaderModule(device, frag_module, nullptr);

    return result == VK_SUCCESS;
}

void Pipeline::Shutdown(VulkanContext& ctx) {
    VkDevice device = ctx.GetDevice();
    if (m_pipeline) vkDestroyPipeline(device, m_pipeline, nullptr);
    if (m_pipeline_layout) vkDestroyPipelineLayout(device, m_pipeline_layout, nullptr);
    if (m_descriptor_pool) vkDestroyDescriptorPool(device, m_descriptor_pool, nullptr);
    if (m_descriptor_set_layout) vkDestroyDescriptorSetLayout(device, m_descriptor_set_layout, nullptr);
    if (m_ubo_buffer) vmaDestroyBuffer(ctx.GetAllocator(), m_ubo_buffer, m_ubo_alloc);
}

void Pipeline::Bind(VkCommandBuffer cmd) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipeline_layout, 0, 1, &m_descriptor_set, 0, nullptr);
}

void Pipeline::PushModel(VkCommandBuffer cmd, const float* model_4x4, const float* color_4) {
    PushConstants pc;
    memcpy(pc.model, model_4x4, sizeof(float) * 16);
    memcpy(pc.color, color_4, sizeof(float) * 4);
    vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(PushConstants), &pc);
}

void Pipeline::UpdateSceneUBO(VulkanContext& ctx, const SceneUBO& ubo) {
    void* mapped;
    vmaMapMemory(ctx.GetAllocator(), m_ubo_alloc, &mapped);
    memcpy(mapped, &ubo, sizeof(SceneUBO));
    vmaUnmapMemory(ctx.GetAllocator(), m_ubo_alloc);
}

} // namespace joltgym
