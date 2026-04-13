#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "core/physics_world.h"
#include "core/articulation.h"
#include "core/state_extractor.h"
#include "mjcf/mjcf_parser.h"
#include "mjcf/mjcf_to_jolt.h"

#include <random>
#include <tuple>
#include <cmath>

namespace py = pybind11;

namespace joltgym {

class HumanoidCore {
public:
    HumanoidCore(const std::string& model_path,
                 float forward_reward_weight = 1.25f,
                 float ctrl_cost_weight = 0.1f,
                 float healthy_reward = 5.0f,
                 float healthy_z_min = 1.0f,
                 float healthy_z_max = 2.0f)
        : m_forward_reward_weight(forward_reward_weight)
        , m_ctrl_cost_weight(ctrl_cost_weight)
        , m_healthy_reward(healthy_reward)
        , m_healthy_z_min(healthy_z_min)
        , m_healthy_z_max(healthy_z_max)
    {
        // Parse MJCF
        MjcfParser parser;
        m_model = parser.Parse(model_path);

        // Build physics world
        m_world.Init();
        MjcfToJolt builder;
        m_articulation = builder.Build(m_model, m_world);

        // Humanoid has free root → skip x,y from qpos (2 elements)
        m_state = std::make_unique<StateExtractor>(m_articulation, &m_world, /*qpos_skip=*/2);
        m_dt = m_model.option.timestep;

        // Allocate observation buffer
        m_obs_dim = m_state->GetObsDim();
        m_act_dim = m_state->GetActionDim();
        m_obs_buffer.resize(m_obs_dim);
    }

    std::tuple<py::array_t<float>, float, bool, bool> step(py::array_t<float> action) {
        auto action_buf = action.unchecked<1>();

        float x_before = m_state->GetRootX();

        // Apply actions and step
        std::vector<float> actions(m_act_dim);
        for (int i = 0; i < m_act_dim; i++) {
            actions[i] = action_buf(i);
        }

        for (int i = 0; i < m_frame_skip; i++) {
            m_articulation->ApplyActions(actions.data(), m_act_dim);
            m_world.Step(m_dt, 1);
        }

        float x_after = m_state->GetRootX();

        // Compute reward
        float dt = m_frame_skip * m_dt;
        m_x_velocity = (x_after - x_before) / dt;
        m_forward_reward_val = m_forward_reward_weight * m_x_velocity;

        float ctrl_cost = 0;
        for (int i = 0; i < m_act_dim; i++) {
            ctrl_cost += actions[i] * actions[i];
        }
        m_ctrl_cost_val = m_ctrl_cost_weight * ctrl_cost;

        // Check healthy
        float root_z = m_state->GetRootZ();
        bool is_healthy = (root_z >= m_healthy_z_min && root_z <= m_healthy_z_max);
        float healthy_reward_val = is_healthy ? m_healthy_reward : 0.0f;

        float reward = m_forward_reward_val + healthy_reward_val - m_ctrl_cost_val;
        m_episode_length++;

        bool terminated = !is_healthy;

        // Extract observation
        m_state->ExtractObs(m_obs_buffer.data());

        // Clip observations to prevent NaN propagation
        for (int i = 0; i < m_obs_dim; i++) {
            float v = m_obs_buffer[i];
            if (std::isnan(v) || std::isinf(v)) v = 0.0f;
            m_obs_buffer[i] = std::clamp(v, -100.0f, 100.0f);
        }

        auto obs = py::array_t<float>(m_obs_dim);
        auto obs_buf = obs.mutable_unchecked<1>();
        for (int i = 0; i < m_obs_dim; i++) {
            obs_buf(i) = m_obs_buffer[i];
        }

        return std::make_tuple(obs, reward, terminated, false);
    }

    py::array_t<float> reset(std::optional<uint32_t> seed = std::nullopt,
                              float noise_scale = 0.005f) {
        if (seed.has_value()) {
            m_rng.seed(seed.value());
        }

        m_world.RestoreSnapshot();

        // Add noise to initial state (smaller for humanoid — it's 3D and sensitive)
        if (noise_scale > 0) {
            auto& body_interface = m_world.GetBodyInterface();
            auto root_id = m_articulation->GetRootBody();

            // Small position noise
            std::uniform_real_distribution<float> uniform(-noise_scale, noise_scale);
            auto pos = body_interface.GetPosition(root_id);
            body_interface.SetPosition(root_id,
                JPH::RVec3(pos.GetX() + uniform(m_rng),
                           pos.GetY() + uniform(m_rng),
                           pos.GetZ() + uniform(m_rng) * 0.1f),
                JPH::EActivation::Activate);

            // Small velocity noise
            std::normal_distribution<float> normal(0.0f, noise_scale);
            body_interface.SetLinearVelocity(root_id,
                JPH::Vec3(normal(m_rng), normal(m_rng), normal(m_rng)));
        }

        m_episode_length = 0;
        m_forward_reward_val = 0;
        m_ctrl_cost_val = 0;
        m_x_velocity = 0;

        // Extract observation
        m_state->ExtractObs(m_obs_buffer.data());
        auto obs = py::array_t<float>(m_obs_dim);
        auto obs_buf = obs.mutable_unchecked<1>();
        for (int i = 0; i < m_obs_dim; i++) {
            obs_buf(i) = m_obs_buffer[i];
        }

        return obs;
    }

    float get_root_x() const { return m_state->GetRootX(); }
    float get_root_z() const { return m_state->GetRootZ(); }
    float get_x_velocity() const { return m_x_velocity; }
    float get_forward_reward() const { return m_forward_reward_val; }
    float get_ctrl_cost() const { return m_ctrl_cost_val; }
    int get_obs_dim() const { return m_obs_dim; }
    int get_action_dim() const { return m_act_dim; }
    int get_episode_length() const { return m_episode_length; }

    void shutdown() {
        // Cleanup
    }

private:
    MjcfModel m_model;
    PhysicsWorld m_world;
    Articulation* m_articulation = nullptr;
    std::unique_ptr<StateExtractor> m_state;

    float m_dt = 0.003f;
    int m_frame_skip = 5;
    float m_forward_reward_weight;
    float m_ctrl_cost_weight;
    float m_healthy_reward;
    float m_healthy_z_min;
    float m_healthy_z_max;

    int m_obs_dim = 0;
    int m_act_dim = 0;
    std::vector<float> m_obs_buffer;

    int m_episode_length = 0;
    float m_forward_reward_val = 0;
    float m_ctrl_cost_val = 0;
    float m_x_velocity = 0;

    std::mt19937 m_rng{42};
};

} // namespace joltgym

void bind_humanoid(py::module_& m) {
    py::class_<joltgym::HumanoidCore>(m, "HumanoidCore")
        .def(py::init<const std::string&, float, float, float, float, float>(),
             py::arg("model_path"),
             py::arg("forward_reward_weight") = 1.25f,
             py::arg("ctrl_cost_weight") = 0.1f,
             py::arg("healthy_reward") = 5.0f,
             py::arg("healthy_z_min") = 1.0f,
             py::arg("healthy_z_max") = 2.0f)
        .def("step", &joltgym::HumanoidCore::step)
        .def("reset", &joltgym::HumanoidCore::reset,
             py::arg("seed") = py::none(),
             py::arg("noise_scale") = 0.005f)
        .def("get_root_x", &joltgym::HumanoidCore::get_root_x)
        .def("get_root_z", &joltgym::HumanoidCore::get_root_z)
        .def("get_x_velocity", &joltgym::HumanoidCore::get_x_velocity)
        .def("get_forward_reward", &joltgym::HumanoidCore::get_forward_reward)
        .def("get_ctrl_cost", &joltgym::HumanoidCore::get_ctrl_cost)
        .def("get_obs_dim", &joltgym::HumanoidCore::get_obs_dim)
        .def("get_action_dim", &joltgym::HumanoidCore::get_action_dim)
        .def("get_episode_length", &joltgym::HumanoidCore::get_episode_length)
        .def("shutdown", &joltgym::HumanoidCore::shutdown);
}
