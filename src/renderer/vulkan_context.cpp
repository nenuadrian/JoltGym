#define VMA_IMPLEMENTATION
#include "vulkan_context.h"
#include <stdexcept>
#include <cstdio>

namespace joltgym {

bool VulkanContext::Init(bool enable_validation, VkSurfaceKHR surface) {
    // Build instance
    auto inst_builder = vkb::InstanceBuilder()
        .set_app_name("JoltGym")
        .set_engine_name("JoltGym Renderer")
        .require_api_version(1, 2, 0);

    if (enable_validation) {
        inst_builder.request_validation_layers()
                    .use_default_debug_messenger();
    }

    // Enable portability on macOS (MoltenVK)
    inst_builder.enable_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

    auto inst_ret = inst_builder.build();
    if (!inst_ret) {
        fprintf(stderr, "Failed to create Vulkan instance: %s\n", inst_ret.error().message().c_str());
        return false;
    }

    m_vkb_instance = inst_ret.value();
    m_instance = m_vkb_instance.instance;
    m_debug_messenger = m_vkb_instance.debug_messenger;

    // Select physical device
    auto phys_selector = vkb::PhysicalDeviceSelector(m_vkb_instance);
    if (surface != VK_NULL_HANDLE) {
        phys_selector.set_surface(surface);
    }
    phys_selector.set_minimum_version(1, 2);

    auto phys_ret = phys_selector.select();
    if (!phys_ret) {
        fprintf(stderr, "Failed to select GPU: %s\n", phys_ret.error().message().c_str());
        return false;
    }

    m_physical_device = phys_ret.value().physical_device;

    // Build logical device
    auto dev_builder = vkb::DeviceBuilder(phys_ret.value());
    auto dev_ret = dev_builder.build();
    if (!dev_ret) {
        fprintf(stderr, "Failed to create logical device: %s\n", dev_ret.error().message().c_str());
        return false;
    }

    auto vkb_device = dev_ret.value();
    m_device = vkb_device.device;

    auto queue_ret = vkb_device.get_queue(vkb::QueueType::graphics);
    if (!queue_ret) return false;
    m_graphics_queue = queue_ret.value();
    m_graphics_queue_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

    // Create VMA allocator
    VmaAllocatorCreateInfo alloc_info = {};
    alloc_info.physicalDevice = m_physical_device;
    alloc_info.device = m_device;
    alloc_info.instance = m_instance;
    alloc_info.vulkanApiVersion = VK_API_VERSION_1_2;

    if (vmaCreateAllocator(&alloc_info, &m_allocator) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create VMA allocator\n");
        return false;
    }

    return true;
}

void VulkanContext::Shutdown() {
    if (m_allocator) {
        vmaDestroyAllocator(m_allocator);
        m_allocator = VK_NULL_HANDLE;
    }
    if (m_device) {
        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }
    if (m_debug_messenger) {
        vkb::destroy_debug_utils_messenger(m_instance, m_debug_messenger);
        m_debug_messenger = VK_NULL_HANDLE;
    }
    if (m_instance) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }
}

VkCommandPool VulkanContext::CreateCommandPool() {
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = m_graphics_queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool pool;
    if (vkCreateCommandPool(m_device, &pool_info, nullptr, &pool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
    return pool;
}

VkCommandBuffer VulkanContext::AllocateCommandBuffer(VkCommandPool pool) {
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(m_device, &alloc_info, &cmd);
    return cmd;
}

void VulkanContext::SubmitAndWait(VkCommandBuffer cmd, VkCommandPool pool) {
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    vkCreateFence(m_device, &fence_info, nullptr, &fence);

    vkQueueSubmit(m_graphics_queue, 1, &submit, fence);
    vkWaitForFences(m_device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(m_device, fence, nullptr);
    vkFreeCommandBuffers(m_device, pool, 1, &cmd);
}

} // namespace joltgym
