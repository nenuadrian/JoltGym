#include "multi_agent_env.h"
#include <cstring>

namespace joltgym {

MultiAgentEnv::MultiAgentEnv(int num_agents, const std::string& model_path,
                             float agent_spacing,
                             float forward_reward_weight,
                             float ctrl_cost_weight)
    : m_num_agents(num_agents)
    , m_forward_reward_weight(forward_reward_weight)
    , m_ctrl_cost_weight(ctrl_cost_weight)
    , m_agent_spacing(agent_spacing)
{
    // Parse model once
    MjcfParser parser;
    m_model = parser.Parse(model_path);
    m_dt = m_model.option.timestep;

    // Single shared world for all agents
    m_world.Init();

    // Build N robots at different Y-offsets (same X start for racing)
    m_agents.resize(num_agents);
    for (int i = 0; i < num_agents; i++) {
        float y_offset = (i - (num_agents - 1) * 0.5f) * agent_spacing;
        JPH::Vec3 offset(0.0f, y_offset, 0.0f);

        MjcfToJolt builder;
        bool first = (i == 0);
        m_agents[i].articulation = builder.Build(m_model, m_world, offset, first);
        m_agents[i].state = std::make_unique<StateExtractor>(
            m_agents[i].articulation, &m_world);
        m_agents[i].obs_buffer.resize(m_agents[i].state->GetObsDim());
        m_agents[i].rng.seed(42 + i);
    }

    m_obs_dim = m_agents[0].state->GetObsDim();
    m_act_dim = m_agents[0].state->GetActionDim();

    // Save initial snapshot for reset
    m_world.SaveSnapshot();
}

void MultiAgentEnv::Step(const float* actions, float* obs_out, float* rewards_out) {
    // Record pre-step positions
    for (int i = 0; i < m_num_agents; i++) {
        m_agents[i].prev_x = m_agents[i].state->GetRootX();
    }

    // Apply actions for all agents
    for (int i = 0; i < m_num_agents; i++) {
        const float* agent_actions = actions + i * m_act_dim;
        m_agents[i].articulation->ApplyActions(agent_actions, m_act_dim);
    }

    // Step the shared world (all agents simulated together)
    for (int f = 0; f < m_frame_skip; f++) {
        // Re-apply actions each sub-step
        for (int i = 0; i < m_num_agents; i++) {
            const float* agent_actions = actions + i * m_act_dim;
            m_agents[i].articulation->ApplyActions(agent_actions, m_act_dim);
        }
        m_world.Step(m_dt, 1);
    }

    // Extract per-agent observations and rewards
    float dt = m_frame_skip * m_dt;
    for (int i = 0; i < m_num_agents; i++) {
        float x_after = m_agents[i].state->GetRootX();
        m_agents[i].x_velocity = (x_after - m_agents[i].prev_x) / dt;
        m_agents[i].forward_reward = m_forward_reward_weight * m_agents[i].x_velocity;

        // Control cost
        const float* agent_actions = actions + i * m_act_dim;
        float ctrl_cost = 0;
        for (int j = 0; j < m_act_dim; j++) {
            ctrl_cost += agent_actions[j] * agent_actions[j];
        }
        m_agents[i].ctrl_cost = m_ctrl_cost_weight * ctrl_cost;

        rewards_out[i] = m_agents[i].forward_reward - m_agents[i].ctrl_cost;

        // Extract observations
        m_agents[i].state->ExtractObs(m_agents[i].obs_buffer.data());
        std::memcpy(obs_out + i * m_obs_dim, m_agents[i].obs_buffer.data(),
                    m_obs_dim * sizeof(float));
    }
}

void MultiAgentEnv::ResetAll(float* obs_out, std::optional<uint32_t> base_seed,
                             float noise_scale) {
    m_world.RestoreSnapshot();

    for (int i = 0; i < m_num_agents; i++) {
        auto& agent = m_agents[i];
        if (base_seed.has_value()) {
            agent.rng.seed(base_seed.value() + i);
        }

        if (noise_scale > 0) {
            auto& bi = m_world.GetBodyInterface();
            auto root_id = agent.articulation->GetRootBody();

            std::uniform_real_distribution<float> uniform(-noise_scale, noise_scale);
            auto pos = bi.GetPosition(root_id);
            bi.SetPosition(root_id,
                JPH::RVec3(pos.GetX() + uniform(agent.rng),
                           pos.GetY(),
                           pos.GetZ() + uniform(agent.rng) * 0.1f),
                JPH::EActivation::Activate);

            std::normal_distribution<float> normal(0.0f, noise_scale * 0.5f);
            bi.SetLinearVelocity(root_id,
                JPH::Vec3(normal(agent.rng), 0, normal(agent.rng)));
        }

        agent.x_velocity = 0;
        agent.forward_reward = 0;
        agent.ctrl_cost = 0;
        agent.prev_x = agent.state->GetRootX();

        agent.state->ExtractObs(agent.obs_buffer.data());
        std::memcpy(obs_out + i * m_obs_dim, agent.obs_buffer.data(),
                    m_obs_dim * sizeof(float));
    }
}

float MultiAgentEnv::GetAgentX(int idx) const {
    return m_agents[idx].state->GetRootX();
}

float MultiAgentEnv::GetAgentXVelocity(int idx) const {
    return m_agents[idx].x_velocity;
}

} // namespace joltgym
