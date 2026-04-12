#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <VkBootstrap.h>

namespace joltgym {

class VulkanContext {
public:
    bool Init(bool enable_validation = true, VkSurfaceKHR surface = VK_NULL_HANDLE);
    void Shutdown();

    VkInstance GetInstance() const { return m_instance; }
    VkPhysicalDevice GetPhysicalDevice() const { return m_physical_device; }
    VkDevice GetDevice() const { return m_device; }
    VkQueue GetGraphicsQueue() const { return m_graphics_queue; }
    uint32_t GetGraphicsQueueFamily() const { return m_graphics_queue_family; }
    VmaAllocator GetAllocator() const { return m_allocator; }
    vkb::Instance& GetVkbInstance() { return m_vkb_instance; }

    VkCommandPool CreateCommandPool();
    VkCommandBuffer AllocateCommandBuffer(VkCommandPool pool);
    void SubmitAndWait(VkCommandBuffer cmd, VkCommandPool pool);

private:
    vkb::Instance m_vkb_instance;
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_graphics_queue = VK_NULL_HANDLE;
    uint32_t m_graphics_queue_family = 0;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debug_messenger = VK_NULL_HANDLE;
};

} // namespace joltgym
