/// @file world_pool.h
/// @brief N parallel physics environments stepped from C++ threads.
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

/// @brief Independent state for a single environment in the pool.
///
/// Each EnvInstance owns its own PhysicsWorld (with single-threaded Jolt),
/// allowing contention-free parallel stepping.
struct EnvInstance {
    PhysicsWorld world;                        ///< Per-env physics system.
    Articulation* articulation = nullptr;      ///< The robot articulation.
    std::unique_ptr<StateExtractor> state;     ///< Observation extractor.
    std::vector<float> obs_buffer;             ///< Pre-allocated observation buffer.
    std::vector<float> action_buffer;          ///< Pre-allocated action buffer.

    float dt = 0.01f;                          ///< Physics timestep (seconds).
    int frame_skip = 5;                        ///< Physics steps per env step.
    float forward_reward_weight = 1.0f;        ///< Weight on forward velocity reward.
    float ctrl_cost_weight = 0.1f;             ///< Weight on control cost penalty.

    int obs_dim = 0;                           ///< Observation dimension.
    int act_dim = 0;                           ///< Action dimension.
    int episode_length = 0;                    ///< Steps in current episode.
    float x_velocity = 0;                      ///< Last computed X velocity.
    float forward_reward = 0;                  ///< Last forward reward.
    float ctrl_cost = 0;                       ///< Last control cost.
    float reward = 0;                          ///< Last total reward.

    std::mt19937 rng{42};                      ///< Per-env RNG for resets.
};

/// @brief Pool of N parallel environments stepped in native C++ threads.
///
/// The MJCF model is parsed once and used to build N independent PhysicsWorld
/// instances, each with its own single-threaded Jolt JobSystem. StepAll()
/// distributes environments across `min(hardware_concurrency, 16)` OS threads
/// via ParallelFor, with the Python GIL released for the entire duration.
///
/// Achieves ~73K env-steps/sec at 256 environments on Apple Silicon.
class WorldPool {
public:
    /// @param num_envs              Number of parallel environments.
    /// @param model_path            Path to the MJCF XML model file.
    /// @param forward_reward_weight Weight on forward velocity reward.
    /// @param ctrl_cost_weight      Weight on control cost penalty.
    WorldPool(int num_envs, const std::string& model_path,
              float forward_reward_weight = 1.0f,
              float ctrl_cost_weight = 0.1f);

    int num_envs() const { return (int)m_envs.size(); } ///< Number of environments.
    int obs_dim() const { return m_obs_dim; }           ///< Observation dimension per env.
    int act_dim() const { return m_act_dim; }           ///< Action dimension per env.

    /// @brief Step all environments in parallel.
    /// @param actions     Flat array of shape `[num_envs * act_dim]`.
    /// @param obs_out     Output observations, shape `[num_envs * obs_dim]`.
    /// @param rewards_out Output rewards, shape `[num_envs]`.
    /// @param dones_out   Output terminal flags, shape `[num_envs]`.
    ///                    True indicates the env was auto-reset.
    void StepAll(const float* actions, float* obs_out,
                 float* rewards_out, bool* dones_out);

    /// @brief Reset all environments in parallel.
    /// @param obs_out   Output observations, shape `[num_envs * obs_dim]`.
    /// @param base_seed Optional seed; environment i gets `base_seed + i`.
    /// @param noise_scale Standard deviation of reset noise.
    void ResetAll(float* obs_out, std::optional<uint32_t> base_seed = std::nullopt,
                  float noise_scale = 0.1f);

    /// @brief Reset a single environment by index.
    void ResetOne(int idx, float* obs_out,
                  std::optional<uint32_t> seed = std::nullopt,
                  float noise_scale = 0.1f);

private:
    void StepEnv(int idx, const float* actions);
    void ResetEnv(int idx, std::optional<uint32_t> seed, float noise_scale);

    std::vector<std::unique_ptr<EnvInstance>> m_envs;
    MjcfModel m_model;
    int m_obs_dim = 0;
    int m_act_dim = 0;
    int m_num_threads = 1;
};

} // namespace joltgym
