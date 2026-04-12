#include "offscreen_renderer.h"
#include <cstring>

namespace joltgym {

OffscreenRenderer::~OffscreenRenderer() {
    Shutdown();
}

bool OffscreenRenderer::Init(uint32_t width, uint32_t height) {
    m_width = width;
    m_height = height;
    std::memset(&m_scene_ubo, 0, sizeof(m_scene_ubo));
    m_scene_ubo.ambient = 0.3f;
    m_scene_ubo.light_color[0] = m_scene_ubo.light_color[1] = m_scene_ubo.light_color[2] = 1.0f;

    if (!m_ctx.Init(false)) return false; // No validation for offscreen perf

    VkDevice device = m_ctx.GetDevice();
    VmaAllocator alloc = m_ctx.GetAllocator();

    // Color image (RGBA8)
    VkImageCreateInfo color_info = {};
    color_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    color_info.imageType = VK_IMAGE_TYPE_2D;
    color_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    color_info.extent = {width, height, 1};
    color_info.mipLevels = 1;
    color_info.arrayLayers = 1;
    color_info.samples = VK_SAMPLE_COUNT_1_BIT;
    color_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo img_alloc = {};
    img_alloc.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    vmaCreateImage(alloc, &color_info, &img_alloc, &m_color_image, &m_color_alloc, nullptr);

    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = m_color_image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(device, &view_info, nullptr, &m_color_view);

    // Depth image
    VkImageCreateInfo depth_info = color_info;
    depth_info.format = VK_FORMAT_D32_SFLOAT;
    depth_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    vmaCreateImage(alloc, &depth_info, &img_alloc, &m_depth_image, &m_depth_alloc, nullptr);

    view_info.image = m_depth_image;
    view_info.format = VK_FORMAT_D32_SFLOAT;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vkCreateImageView(device, &view_info, nullptr, &m_depth_view);

    // Readback buffer
    VkBufferCreateInfo buf_info = {};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = width * height * 4;
    buf_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo buf_alloc = {};
    buf_alloc.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
    vmaCreateBuffer(alloc, &buf_info, &buf_alloc, &m_readback_buffer, &m_readback_alloc, nullptr);

    // Render pass
    VkAttachmentDescription attachments[2] = {};
    attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    attachments[1].format = VK_FORMAT_D32_SFLOAT;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference color_ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depth_ref = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;
    subpass.pDepthStencilAttachment = &depth_ref;

    VkRenderPassCreateInfo rp_info = {};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.attachmentCount = 2;
    rp_info.pAttachments = attachments;
    rp_info.subpassCount = 1;
    rp_info.pSubpasses = &subpass;
    vkCreateRenderPass(device, &rp_info, nullptr, &m_render_pass);

    // Framebuffer
    VkImageView views[] = {m_color_view, m_depth_view};
    VkFramebufferCreateInfo fb_info = {};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass = m_render_pass;
    fb_info.attachmentCount = 2;
    fb_info.pAttachments = views;
    fb_info.width = width;
    fb_info.height = height;
    fb_info.layers = 1;
    vkCreateFramebuffer(device, &fb_info, nullptr, &m_framebuffer);

    // Command pool and buffer
    m_cmd_pool = m_ctx.CreateCommandPool();

    // Init pipeline and meshes
    m_pipeline.Init(m_ctx, m_render_pass);
    m_meshes.Init(m_ctx);

    return true;
}

void OffscreenRenderer::BeginFrame() {
    m_cmd = m_ctx.AllocateCommandBuffer(m_cmd_pool);

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_cmd, &begin);

    VkClearValue clears[2] = {};
    clears[0].color = {{0.15f, 0.15f, 0.2f, 1.0f}};
    clears[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rp_begin = {};
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.renderPass = m_render_pass;
    rp_begin.framebuffer = m_framebuffer;
    rp_begin.renderArea.extent = {m_width, m_height};
    rp_begin.clearValueCount = 2;
    rp_begin.pClearValues = clears;
    vkCmdBeginRenderPass(m_cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp = {};
    vp.width = (float)m_width;
    vp.height = (float)m_height;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(m_cmd, 0, 1, &vp);

    VkRect2D scissor = {};
    scissor.extent = {m_width, m_height};
    vkCmdSetScissor(m_cmd, 0, 1, &scissor);

    m_pipeline.UpdateSceneUBO(m_ctx, m_scene_ubo);
    m_pipeline.Bind(m_cmd);
}

void OffscreenRenderer::DrawPrimitive(const RenderTransform& transform) {
    auto& mesh = m_meshes.GetMesh(transform.type);
    m_pipeline.PushModel(m_cmd, transform.model, transform.color);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(m_cmd, 0, 1, &mesh.vertex_buffer, &offset);
    vkCmdBindIndexBuffer(m_cmd, mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(m_cmd, mesh.index_count, 1, 0, 0, 0);
}

void OffscreenRenderer::EndFrame() {
    vkCmdEndRenderPass(m_cmd);

    // Copy color image to readback buffer
    VkBufferImageCopy region = {};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {m_width, m_height, 1};
    vkCmdCopyImageToBuffer(m_cmd, m_color_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           m_readback_buffer, 1, &region);

    vkEndCommandBuffer(m_cmd);
    m_ctx.SubmitAndWait(m_cmd, m_cmd_pool);
}

std::vector<uint8_t> OffscreenRenderer::GetFrameBuffer(uint32_t& width, uint32_t& height) {
    width = m_width;
    height = m_height;

    std::vector<uint8_t> result(m_width * m_height * 3); // RGB

    void* mapped;
    vmaMapMemory(m_ctx.GetAllocator(), m_readback_alloc, &mapped);

    // Convert RGBA -> RGB
    uint8_t* src = (uint8_t*)mapped;
    for (uint32_t i = 0; i < m_width * m_height; i++) {
        result[i * 3 + 0] = src[i * 4 + 0];
        result[i * 3 + 1] = src[i * 4 + 1];
        result[i * 3 + 2] = src[i * 4 + 2];
    }

    vmaUnmapMemory(m_ctx.GetAllocator(), m_readback_alloc);
    return result;
}

void OffscreenRenderer::SetViewMatrix(const float* v) { std::memcpy(m_scene_ubo.view, v, 64); }
void OffscreenRenderer::SetProjectionMatrix(const float* p) { std::memcpy(m_scene_ubo.proj, p, 64); }
void OffscreenRenderer::SetLightDirection(float x, float y, float z) {
    m_scene_ubo.light_dir[0] = x; m_scene_ubo.light_dir[1] = y; m_scene_ubo.light_dir[2] = z;
}
void OffscreenRenderer::SetCameraPosition(float x, float y, float z) {
    m_scene_ubo.camera_pos[0] = x; m_scene_ubo.camera_pos[1] = y; m_scene_ubo.camera_pos[2] = z;
}

void OffscreenRenderer::Shutdown() {
    if (m_ctx.GetDevice()) {
        vkDeviceWaitIdle(m_ctx.GetDevice());
        m_pipeline.Shutdown(m_ctx);
        m_meshes.Shutdown(m_ctx);
        // Destroy other resources
        VkDevice device = m_ctx.GetDevice();
        if (m_framebuffer) vkDestroyFramebuffer(device, m_framebuffer, nullptr);
        if (m_render_pass) vkDestroyRenderPass(device, m_render_pass, nullptr);
        if (m_color_view) vkDestroyImageView(device, m_color_view, nullptr);
        if (m_depth_view) vkDestroyImageView(device, m_depth_view, nullptr);
        if (m_color_image) vmaDestroyImage(m_ctx.GetAllocator(), m_color_image, m_color_alloc);
        if (m_depth_image) vmaDestroyImage(m_ctx.GetAllocator(), m_depth_image, m_depth_alloc);
        if (m_readback_buffer) vmaDestroyBuffer(m_ctx.GetAllocator(), m_readback_buffer, m_readback_alloc);
        if (m_cmd_pool) vkDestroyCommandPool(device, m_cmd_pool, nullptr);
    }
    m_ctx.Shutdown();
}

} // namespace joltgym
