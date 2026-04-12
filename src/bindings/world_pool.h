#pragma once

#include "core/physics_world.h"
#include "core/articulation.h"
#include "core/state_extractor.h"
#include "mjcf/mjcf_parser.h"
#include "mjcf/mjcf_to_jolt.h"

#include <vector>
#include <memory>
#include <random>
#include <thread>

namespace joltgym {

// Per-environment state (no Python/pybind11 dependency)
struct EnvInstance {
    PhysicsWorld world;
    Articulation* articulation = nullptr;
    std::unique_ptr<StateExtractor> state;
    std::vector<float> obs_buffer;
    std::vector<float> action_buffer;

    float dt = 0.01f;
    int frame_skip = 5;
    float forward_reward_weight = 1.0f;
    float ctrl_cost_weight = 0.1f;

    int obs_dim = 0;
    int act_dim = 0;
    int episode_length = 0;
    float x_velocity = 0;
    float forward_reward = 0;
    float ctrl_cost = 0;
    float reward = 0;

    std::mt19937 rng{42};
};

class WorldPool {
public:
    WorldPool(int num_envs, const std::string& model_path,
              float forward_reward_weight = 1.0f,
              float ctrl_cost_weight = 0.1f);

    int num_envs() const { return (int)m_envs.size(); }
    int obs_dim() const { return m_obs_dim; }
    int act_dim() const { return m_act_dim; }

    // Step all environments in parallel.
    // actions: flat array of size [num_envs * act_dim]
    // obs_out: flat array of size [num_envs * obs_dim]
    // rewards_out: array of size [num_envs]
    // dones_out: array of size [num_envs] (auto-reset flag)
    void StepAll(const float* actions, float* obs_out,
                 float* rewards_out, bool* dones_out);

    // Reset all environments in parallel.
    // obs_out: flat array of size [num_envs * obs_dim]
    void ResetAll(float* obs_out, std::optional<uint32_t> base_seed = std::nullopt,
                  float noise_scale = 0.1f);

    // Reset a single environment.
    void ResetOne(int idx, float* obs_out,
                  std::optional<uint32_t> seed = std::nullopt,
                  float noise_scale = 0.1f);

private:
    void StepEnv(int idx, const float* actions);
    void ResetEnv(int idx, std::optional<uint32_t> seed, float noise_scale);

    std::vector<std::unique_ptr<EnvInstance>> m_envs;
    MjcfModel m_model; // Parsed once, shared for building
    int m_obs_dim = 0;
    int m_act_dim = 0;
    int m_num_threads = 1;
};

} // namespace joltgym
