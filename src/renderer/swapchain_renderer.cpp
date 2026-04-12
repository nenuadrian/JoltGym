#include "swapchain_renderer.h"
#include "imgui_layer.h"
#include <cstring>
#include <cstdio>

namespace joltgym {

SwapchainRenderer::SwapchainRenderer() {
    std::memset(&m_scene_ubo, 0, sizeof(m_scene_ubo));
    m_scene_ubo.ambient = 0.3f;
    m_scene_ubo.light_color[0] = m_scene_ubo.light_color[1] = m_scene_ubo.light_color[2] = 1.0f;
    m_scene_ubo.light_dir[0] = -0.5f;
    m_scene_ubo.light_dir[1] = -0.3f;
    m_scene_ubo.light_dir[2] = -1.0f;
}

SwapchainRenderer::~SwapchainRenderer() {
    Shutdown();
}

bool SwapchainRenderer::Init(uint32_t width, uint32_t height) {
    m_width = width;
    m_height = height;

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return false;
    }

    m_window = SDL_CreateWindow("JoltGym - HalfCheetah",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        width, height, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!m_window) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return false;
    }

    // Create Vulkan surface from SDL window
    if (!SDL_Vulkan_CreateSurface(m_window, m_ctx.GetInstance(), &m_surface)) {
        // Need to init instance first, then surface, then the rest
        // Re-init with surface
    }

    // Init Vulkan context — need surface before physical device selection
    // First create instance for SDL surface
    auto inst_builder = vkb::InstanceBuilder()
        .set_app_name("JoltGym")
        .require_api_version(1, 2, 0)
        .request_validation_layers()
        .use_default_debug_messenger()
        .enable_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

    auto inst_ret = inst_builder.build();
    if (!inst_ret) {
        fprintf(stderr, "Failed to create Vulkan instance\n");
        return false;
    }

    auto vkb_instance = inst_ret.value();

    if (!SDL_Vulkan_CreateSurface(m_window, vkb_instance.instance, &m_surface)) {
        fprintf(stderr, "SDL_Vulkan_CreateSurface failed: %s\n", SDL_GetError());
        return false;
    }

    // Select physical device with surface
    auto phys_ret = vkb::PhysicalDeviceSelector(vkb_instance)
        .set_surface(m_surface)
        .set_minimum_version(1, 2)
        .select();
    if (!phys_ret) {
        fprintf(stderr, "Failed to select GPU\n");
        return false;
    }

    auto dev_ret = vkb::DeviceBuilder(phys_ret.value()).build();
    if (!dev_ret) {
        fprintf(stderr, "Failed to create device\n");
        return false;
    }

    auto vkb_device = dev_ret.value();

    // Manually set up the context
    // We need a slightly different init path since we created instance/device ourselves
    // Store what we need
    VkDevice device = vkb_device.device;
    VkPhysicalDevice phys_device = phys_ret.value().physical_device;
    VkQueue graphics_queue = vkb_device.get_queue(vkb::QueueType::graphics).value();
    uint32_t queue_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

    // Create VMA allocator
    VmaAllocatorCreateInfo alloc_info = {};
    alloc_info.physicalDevice = phys_device;
    alloc_info.device = device;
    alloc_info.instance = vkb_instance.instance;
    alloc_info.vulkanApiVersion = VK_API_VERSION_1_2;

    VmaAllocator allocator;
    vmaCreateAllocator(&alloc_info, &allocator);

    // Store in a simple struct for now — the context owns everything
    // We'll use device/queue/allocator directly
    m_ctx.~VulkanContext(); // Reset
    new (&m_ctx) VulkanContext();
    // We need to store these — let's use the device directly
    // For simplicity, store vkb objects and raw handles

    // Actually let's just keep the raw handles we need
    m_vkb_swapchain = {}; // Will be created below

    // Store the instance/device for cleanup
    // We'll track them through the swapchain path

    // Create swapchain
    auto sc_ret = vkb::SwapchainBuilder(vkb_device)
        .set_desired_format({VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .build();

    if (!sc_ret) {
        fprintf(stderr, "Failed to create swapchain\n");
        return false;
    }

    m_vkb_swapchain = sc_ret.value();
    m_swapchain = m_vkb_swapchain.swapchain;
    m_swapchain_images = m_vkb_swapchain.get_images().value();
    m_swapchain_views = m_vkb_swapchain.get_image_views().value();
    m_swapchain_format = m_vkb_swapchain.image_format;
    m_swapchain_extent = m_vkb_swapchain.extent;

    // Create depth buffer
    VkImageCreateInfo depth_info = {};
    depth_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    depth_info.imageType = VK_IMAGE_TYPE_2D;
    depth_info.format = VK_FORMAT_D32_SFLOAT;
    depth_info.extent = {m_swapchain_extent.width, m_swapchain_extent.height, 1};
    depth_info.mipLevels = 1;
    depth_info.arrayLayers = 1;
    depth_info.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VmaAllocationCreateInfo depth_alloc_info = {};
    depth_alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(allocator, &depth_info, &depth_alloc_info,
                   &m_depth_image, &m_depth_alloc, nullptr);

    VkImageViewCreateInfo depth_view_info = {};
    depth_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depth_view_info.image = m_depth_image;
    depth_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depth_view_info.format = VK_FORMAT_D32_SFLOAT;
    depth_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depth_view_info.subresourceRange.levelCount = 1;
    depth_view_info.subresourceRange.layerCount = 1;
    vkCreateImageView(device, &depth_view_info, nullptr, &m_depth_view);

    // Create render pass
    VkAttachmentDescription color_attachment = {};
    color_attachment.format = m_swapchain_format;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth_attachment = {};
    depth_attachment.format = VK_FORMAT_D32_SFLOAT;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference color_ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depth_ref = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;
    subpass.pDepthStencilAttachment = &depth_ref;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkAttachmentDescription attachments[] = {color_attachment, depth_attachment};
    VkRenderPassCreateInfo rp_info = {};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.attachmentCount = 2;
    rp_info.pAttachments = attachments;
    rp_info.subpassCount = 1;
    rp_info.pSubpasses = &subpass;
    rp_info.dependencyCount = 1;
    rp_info.pDependencies = &dependency;
    vkCreateRenderPass(device, &rp_info, nullptr, &m_render_pass);

    // Create framebuffers
    m_framebuffers.resize(m_swapchain_images.size());
    for (size_t i = 0; i < m_swapchain_images.size(); i++) {
        VkImageView views[] = {m_swapchain_views[i], m_depth_view};
        VkFramebufferCreateInfo fb_info = {};
        fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass = m_render_pass;
        fb_info.attachmentCount = 2;
        fb_info.pAttachments = views;
        fb_info.width = m_swapchain_extent.width;
        fb_info.height = m_swapchain_extent.height;
        fb_info.layers = 1;
        vkCreateFramebuffer(device, &fb_info, nullptr, &m_framebuffers[i]);
    }

    // Command pool and buffers
    VkCommandPoolCreateInfo pool_ci = {};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.queueFamilyIndex = queue_family;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device, &pool_ci, nullptr, &m_cmd_pool);

    m_cmd_buffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo cmd_alloc = {};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = m_cmd_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    vkAllocateCommandBuffers(device, &cmd_alloc, m_cmd_buffers.data());

    // Sync objects
    VkSemaphoreCreateInfo sem_info = {};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkCreateSemaphore(device, &sem_info, nullptr, &m_image_available[i]);
        vkCreateSemaphore(device, &sem_info, nullptr, &m_render_finished[i]);
        vkCreateFence(device, &fence_info, nullptr, &m_in_flight[i]);
    }

    // Init pipeline and meshes
    // Note: We need to pass the context with the device to pipeline
    // For now, set up a minimal context wrapper
    // TODO: Clean up context ownership
    m_pipeline.Init(m_ctx, m_render_pass);
    m_meshes.Init(m_ctx);

    return true;
}

void SwapchainRenderer::PollEvents() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            m_should_close = true;
        }
        if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) {
            m_should_close = true;
        }
    }
}

void SwapchainRenderer::BeginFrame() {
    VkDevice device = m_ctx.GetDevice();

    vkWaitForFences(device, 1, &m_in_flight[m_current_frame], VK_TRUE, UINT64_MAX);

    auto result = vkAcquireNextImageKHR(device, m_swapchain, UINT64_MAX,
        m_image_available[m_current_frame], VK_NULL_HANDLE, &m_image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        // TODO: Recreate swapchain
        return;
    }

    vkResetFences(device, 1, &m_in_flight[m_current_frame]);
    vkResetCommandBuffer(m_cmd_buffers[m_current_frame], 0);

    VkCommandBuffer cmd = m_cmd_buffers[m_current_frame];
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &begin_info);

    VkClearValue clear_values[2] = {};
    clear_values[0].color = {{0.15f, 0.15f, 0.2f, 1.0f}};
    clear_values[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rp_begin = {};
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.renderPass = m_render_pass;
    rp_begin.framebuffer = m_framebuffers[m_image_index];
    rp_begin.renderArea.extent = m_swapchain_extent;
    rp_begin.clearValueCount = 2;
    rp_begin.pClearValues = clear_values;
    vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = {};
    viewport.width = (float)m_swapchain_extent.width;
    viewport.height = (float)m_swapchain_extent.height;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.extent = m_swapchain_extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Update scene UBO
    m_pipeline.UpdateSceneUBO(m_ctx, m_scene_ubo);
    m_pipeline.Bind(cmd);
}

void SwapchainRenderer::DrawPrimitive(const RenderTransform& transform) {
    VkCommandBuffer cmd = m_cmd_buffers[m_current_frame];
    auto& mesh = m_meshes.GetMesh(transform.type);

    m_pipeline.PushModel(cmd, transform.model, transform.color);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertex_buffer, &offset);
    vkCmdBindIndexBuffer(cmd, mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, mesh.index_count, 1, 0, 0, 0);
}

void SwapchainRenderer::EndFrame() {
    VkCommandBuffer cmd = m_cmd_buffers[m_current_frame];

    // Render ImGui if available
    if (m_imgui_layer) {
        m_imgui_layer->Render(cmd);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &m_image_available[m_current_frame];
    submit.pWaitDstStageMask = &wait_stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &m_render_finished[m_current_frame];

    vkQueueSubmit(m_ctx.GetGraphicsQueue(), 1, &submit, m_in_flight[m_current_frame]);

    VkPresentInfoKHR present = {};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &m_render_finished[m_current_frame];
    present.swapchainCount = 1;
    present.pSwapchains = &m_swapchain;
    present.pImageIndices = &m_image_index;
    vkQueuePresentKHR(m_ctx.GetGraphicsQueue(), &present);

    m_current_frame = (m_current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void SwapchainRenderer::SetViewMatrix(const float* view_4x4) {
    std::memcpy(m_scene_ubo.view, view_4x4, sizeof(float) * 16);
}

void SwapchainRenderer::SetProjectionMatrix(const float* proj_4x4) {
    std::memcpy(m_scene_ubo.proj, proj_4x4, sizeof(float) * 16);
}

void SwapchainRenderer::SetLightDirection(float x, float y, float z) {
    m_scene_ubo.light_dir[0] = x;
    m_scene_ubo.light_dir[1] = y;
    m_scene_ubo.light_dir[2] = z;
}

void SwapchainRenderer::SetCameraPosition(float x, float y, float z) {
    m_scene_ubo.camera_pos[0] = x;
    m_scene_ubo.camera_pos[1] = y;
    m_scene_ubo.camera_pos[2] = z;
}

void SwapchainRenderer::Shutdown() {
    if (m_ctx.GetDevice()) {
        vkDeviceWaitIdle(m_ctx.GetDevice());
        m_pipeline.Shutdown(m_ctx);
        m_meshes.Shutdown(m_ctx);
    }
    if (m_window) {
        SDL_DestroyWindow(m_window);
        m_window = nullptr;
        SDL_Quit();
    }
}

} // namespace joltgym
