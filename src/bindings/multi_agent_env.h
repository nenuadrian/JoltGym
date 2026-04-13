#pragma once

#include "core/physics_world.h"
#include "core/articulation.h"
#include "core/state_extractor.h"
#include "mjcf/mjcf_parser.h"
#include "mjcf/mjcf_to_jolt.h"

#include <vector>
#include <memory>
#include <random>

namespace joltgym {

// Multiple robots in ONE shared PhysicsWorld — they can physically interact.
class MultiAgentEnv {
public:
    MultiAgentEnv(int num_agents, const std::string& model_path,
                  float agent_spacing = 3.0f,
                  float forward_reward_weight = 1.0f,
                  float ctrl_cost_weight = 0.1f);

    int num_agents() const { return m_num_agents; }
    int obs_dim() const { return m_obs_dim; }
    int act_dim() const { return m_act_dim; }

    // Step the shared world. All agents act simultaneously.
    // actions: [num_agents * act_dim]
    // obs_out: [num_agents * obs_dim]
    // rewards_out: [num_agents]
    void Step(const float* actions, float* obs_out, float* rewards_out);

    // Reset all agents to initial state.
    void ResetAll(float* obs_out, std::optional<uint32_t> base_seed = std::nullopt,
                  float noise_scale = 0.1f);

    // Get per-agent stats
    float GetAgentX(int idx) const;
    float GetAgentXVelocity(int idx) const;

private:
    struct AgentState {
        Articulation* articulation = nullptr;
        std::unique_ptr<StateExtractor> state;
        std::vector<float> obs_buffer;
        float x_velocity = 0;
        float forward_reward = 0;
        float ctrl_cost = 0;
        float prev_x = 0;
        std::mt19937 rng{42};
    };

    PhysicsWorld m_world;
    MjcfModel m_model;
    std::vector<AgentState> m_agents;
    int m_num_agents;
    int m_obs_dim;
    int m_act_dim;
    float m_dt;
    int m_frame_skip = 5;
    float m_forward_reward_weight;
    float m_ctrl_cost_weight;
    float m_agent_spacing;
};

} // namespace joltgym
