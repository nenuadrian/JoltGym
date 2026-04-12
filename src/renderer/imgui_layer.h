#pragma once

#include "vulkan_context.h"
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_vulkan.h>
#include <SDL.h>

namespace joltgym {

class Articulation;

struct SimulationStats {
    float reward = 0;
    float forward_reward = 0;
    float ctrl_cost = 0;
    int episode_length = 0;
    float root_x = 0;
    float root_x_vel = 0;
    float sim_speed = 1.0f;
    bool paused = false;
    bool step_once = false;
    bool reset_requested = false;
    float joint_angles[16] = {};
    float joint_velocities[16] = {};
    int num_joints = 0;
    const char* joint_names[16] = {};
};

class ImGuiLayer {
public:
    bool Init(VulkanContext& ctx, SDL_Window* window, VkRenderPass render_pass,
              uint32_t image_count);
    void Shutdown();

    void NewFrame();
    void BuildUI(SimulationStats& stats);
    void Render(VkCommandBuffer cmd);

    void ProcessEvent(const SDL_Event& event);

private:
    VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
    VulkanContext* m_ctx = nullptr;
};

} // namespace joltgym
