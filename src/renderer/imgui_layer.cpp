#include "imgui_layer.h"
#include <cstdio>

namespace joltgym {

bool ImGuiLayer::Init(VulkanContext& ctx, SDL_Window* window, VkRenderPass render_pass,
                       uint32_t image_count) {
    m_ctx = &ctx;

    // Create descriptor pool for ImGui
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
    };

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = 3;
    pool_info.pPoolSizes = pool_sizes;
    vkCreateDescriptorPool(ctx.GetDevice(), &pool_info, nullptr, &m_descriptor_pool);

    // Init ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    auto& style = ImGui::GetStyle();
    style.WindowRounding = 5.0f;
    style.FrameRounding = 3.0f;
    style.Alpha = 0.95f;

    ImGui_ImplSDL2_InitForVulkan(window);

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = ctx.GetInstance();
    init_info.PhysicalDevice = ctx.GetPhysicalDevice();
    init_info.Device = ctx.GetDevice();
    init_info.QueueFamily = ctx.GetGraphicsQueueFamily();
    init_info.Queue = ctx.GetGraphicsQueue();
    init_info.DescriptorPool = m_descriptor_pool;
    init_info.MinImageCount = image_count;
    init_info.ImageCount = image_count;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.RenderPass = render_pass;

    ImGui_ImplVulkan_Init(&init_info);

    return true;
}

void ImGuiLayer::NewFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::BuildUI(SimulationStats& stats) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 400), ImGuiCond_FirstUseEver);

    ImGui::Begin("JoltGym - HalfCheetah");

    // Simulation controls
    ImGui::SeparatorText("Controls");
    if (ImGui::Button(stats.paused ? "Resume" : "Pause")) {
        stats.paused = !stats.paused;
    }
    ImGui::SameLine();
    if (ImGui::Button("Step")) {
        stats.step_once = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        stats.reset_requested = true;
    }
    ImGui::SliderFloat("Speed", &stats.sim_speed, 0.1f, 5.0f, "%.1fx");

    // Episode info
    ImGui::SeparatorText("Episode");
    ImGui::Text("Step: %d", stats.episode_length);
    ImGui::Text("Reward: %.3f", stats.reward);
    ImGui::Text("Forward: %.3f", stats.forward_reward);
    ImGui::Text("Ctrl Cost: %.3f", stats.ctrl_cost);

    // Root state
    ImGui::SeparatorText("Root Body");
    ImGui::Text("X Position: %.3f", stats.root_x);
    ImGui::Text("X Velocity: %.3f", stats.root_x_vel);

    // Joint states
    if (stats.num_joints > 0) {
        ImGui::SeparatorText("Joints");
        for (int i = 0; i < stats.num_joints; i++) {
            ImGui::Text("%-8s  angle=%.2f  vel=%.2f",
                        stats.joint_names[i] ? stats.joint_names[i] : "?",
                        stats.joint_angles[i],
                        stats.joint_velocities[i]);
        }
    }

    ImGui::End();
}

void ImGuiLayer::Render(VkCommandBuffer cmd) {
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

void ImGuiLayer::ProcessEvent(const SDL_Event& event) {
    ImGui_ImplSDL2_ProcessEvent(&event);
}

void ImGuiLayer::Shutdown() {
    if (m_ctx && m_ctx->GetDevice()) {
        vkDeviceWaitIdle(m_ctx->GetDevice());
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();
        if (m_descriptor_pool) {
            vkDestroyDescriptorPool(m_ctx->GetDevice(), m_descriptor_pool, nullptr);
        }
    }
}

} // namespace joltgym
