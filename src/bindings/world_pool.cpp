#include "world_pool.h"
#include <algorithm>
#include <cstring>
#include <functional>

namespace joltgym {

WorldPool::WorldPool(int num_envs, const std::string& model_path,
                     float forward_reward_weight, float ctrl_cost_weight) {
    // Parse model once
    MjcfParser parser;
    m_model = parser.Parse(model_path);

    // Each world uses JobSystemSingleThreaded (no internal threading), so we
    // can safely step them in parallel from our own thread pool.
    m_num_threads = std::clamp((int)std::thread::hardware_concurrency(), 1, 16);

    // Build N independent environments
    m_envs.resize(num_envs);
    for (int i = 0; i < num_envs; i++) {
        auto env = std::make_unique<EnvInstance>();
        env->forward_reward_weight = forward_reward_weight;
        env->ctrl_cost_weight = ctrl_cost_weight;

        env->world.Init(2048, 4096, 2048, /*single_threaded=*/true);
        MjcfToJolt builder;
        env->articulation = builder.Build(m_model, env->world);

        env->state = std::make_unique<StateExtractor>(env->articulation, &env->world);
        env->dt = m_model.option.timestep;
        env->obs_dim = env->state->GetObsDim();
        env->act_dim = env->state->GetActionDim();
        env->obs_buffer.resize(env->obs_dim);
        env->action_buffer.resize(env->act_dim);
        env->rng.seed(42 + i);

        m_envs[i] = std::move(env);
    }

    m_obs_dim = m_envs[0]->obs_dim;
    m_act_dim = m_envs[0]->act_dim;
}

void WorldPool::StepEnv(int idx, const float* actions) {
    auto& env = *m_envs[idx];

    float x_before = env.state->GetRootX();

    const float* env_actions = actions + idx * m_act_dim;
    for (int i = 0; i < m_act_dim; i++) {
        env.action_buffer[i] = env_actions[i];
    }

    for (int i = 0; i < env.frame_skip; i++) {
        env.articulation->ApplyActions(env.action_buffer.data(), m_act_dim);
        env.world.Step(env.dt, 1);
    }

    float x_after = env.state->GetRootX();

    float dt = env.frame_skip * env.dt;
    env.x_velocity = (x_after - x_before) / dt;
    env.forward_reward = env.forward_reward_weight * env.x_velocity;

    float ctrl_cost = 0;
    for (int i = 0; i < m_act_dim; i++) {
        ctrl_cost += env.action_buffer[i] * env.action_buffer[i];
    }
    env.ctrl_cost = env.ctrl_cost_weight * ctrl_cost;
    env.reward = env.forward_reward - env.ctrl_cost;
    env.episode_length++;

    env.state->ExtractObs(env.obs_buffer.data());
}

void WorldPool::ResetEnv(int idx, std::optional<uint32_t> seed, float noise_scale) {
    auto& env = *m_envs[idx];

    if (seed.has_value()) {
        env.rng.seed(seed.value());
    }

    env.world.RestoreSnapshot();

    if (noise_scale > 0) {
        auto& body_interface = env.world.GetBodyInterface();
        auto root_id = env.articulation->GetRootBody();

        std::uniform_real_distribution<float> uniform(-noise_scale, noise_scale);
        auto pos = body_interface.GetPosition(root_id);
        body_interface.SetPosition(root_id,
            JPH::RVec3(pos.GetX() + uniform(env.rng),
                       pos.GetY(),
                       pos.GetZ() + uniform(env.rng) * 0.1f),
            JPH::EActivation::Activate);

        std::normal_distribution<float> normal(0.0f, noise_scale);
        body_interface.SetLinearVelocity(root_id,
            JPH::Vec3(normal(env.rng), 0, normal(env.rng)));
    }

    env.episode_length = 0;
    env.forward_reward = 0;
    env.ctrl_cost = 0;
    env.x_velocity = 0;
    env.reward = 0;

    env.state->ExtractObs(env.obs_buffer.data());
}

// Chunk work across a fixed number of threads to avoid oversubscription
static void ParallelFor(int n, int num_threads, std::function<void(int)> fn) {
    if (num_threads <= 1 || n <= 1) {
        for (int i = 0; i < n; i++) fn(i);
        return;
    }

    int threads_to_use = std::min(num_threads, n);
    std::vector<std::thread> threads;
    threads.reserve(threads_to_use);

    int chunk = (n + threads_to_use - 1) / threads_to_use;
    for (int t = 0; t < threads_to_use; t++) {
        int begin = t * chunk;
        int end = std::min(begin + chunk, n);
        if (begin >= end) break;
        threads.emplace_back([begin, end, &fn]() {
            for (int i = begin; i < end; i++) fn(i);
        });
    }
    for (auto& t : threads) t.join();
}

void WorldPool::StepAll(const float* actions, float* obs_out,
                        float* rewards_out, bool* dones_out) {
    int n = (int)m_envs.size();

    ParallelFor(n, m_num_threads, [this, actions](int i) {
        StepEnv(i, actions);
    });

    for (int i = 0; i < n; i++) {
        auto& env = *m_envs[i];
        std::memcpy(obs_out + i * m_obs_dim, env.obs_buffer.data(),
                    m_obs_dim * sizeof(float));
        rewards_out[i] = env.reward;
        dones_out[i] = false;
    }
}

void WorldPool::ResetAll(float* obs_out, std::optional<uint32_t> base_seed,
                         float noise_scale) {
    int n = (int)m_envs.size();

    ParallelFor(n, m_num_threads, [this, base_seed, noise_scale](int i) {
        std::optional<uint32_t> seed;
        if (base_seed.has_value()) seed = base_seed.value() + i;
        ResetEnv(i, seed, noise_scale);
    });

    for (int i = 0; i < n; i++) {
        std::memcpy(obs_out + i * m_obs_dim, m_envs[i]->obs_buffer.data(),
                    m_obs_dim * sizeof(float));
    }
}

void WorldPool::ResetOne(int idx, float* obs_out,
                         std::optional<uint32_t> seed, float noise_scale) {
    ResetEnv(idx, seed, noise_scale);
    std::memcpy(obs_out, m_envs[idx]->obs_buffer.data(),
                m_obs_dim * sizeof(float));
}

} // namespace joltgym
