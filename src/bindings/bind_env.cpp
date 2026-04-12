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

namespace py = pybind11;

namespace joltgym {

class HalfCheetahCore {
public:
    HalfCheetahCore(const std::string& model_path,
                    float forward_reward_weight = 1.0f,
                    float ctrl_cost_weight = 0.1f)
        : m_forward_reward_weight(forward_reward_weight)
        , m_ctrl_cost_weight(ctrl_cost_weight)
    {
        // Parse MJCF
        MjcfParser parser;
        m_model = parser.Parse(model_path);

        // Build physics world
        m_world.Init();
        MjcfToJolt builder;
        m_articulation = builder.Build(m_model, m_world);

        m_state = std::make_unique<StateExtractor>(m_articulation, &m_world);
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
        m_forward_reward = m_forward_reward_weight * m_x_velocity;

        float ctrl_cost = 0;
        for (int i = 0; i < m_act_dim; i++) {
            ctrl_cost += actions[i] * actions[i];
        }
        m_ctrl_cost = m_ctrl_cost_weight * ctrl_cost;

        float reward = m_forward_reward - m_ctrl_cost;
        m_episode_length++;

        // Extract observation
        m_state->ExtractObs(m_obs_buffer.data());

        auto obs = py::array_t<float>(m_obs_dim);
        auto obs_buf = obs.mutable_unchecked<1>();
        for (int i = 0; i < m_obs_dim; i++) {
            obs_buf(i) = m_obs_buffer[i];
        }

        return std::make_tuple(obs, reward, false, false);
    }

    py::array_t<float> reset(std::optional<uint32_t> seed = std::nullopt,
                              float noise_scale = 0.1f) {
        if (seed.has_value()) {
            m_rng.seed(seed.value());
        }

        m_world.RestoreSnapshot();

        // Add noise to initial state
        if (noise_scale > 0) {
            auto& body_interface = m_world.GetBodyInterface();
            auto root_id = m_articulation->GetRootBody();

            // Small position noise on root
            std::uniform_real_distribution<float> uniform(-noise_scale, noise_scale);
            auto pos = body_interface.GetPosition(root_id);
            body_interface.SetPosition(root_id,
                JPH::RVec3(pos.GetX() + uniform(m_rng),
                           pos.GetY(),
                           pos.GetZ() + uniform(m_rng) * 0.1f),
                JPH::EActivation::Activate);

            // Small velocity noise
            std::normal_distribution<float> normal(0.0f, noise_scale);
            body_interface.SetLinearVelocity(root_id,
                JPH::Vec3(normal(m_rng), 0, normal(m_rng)));
        }

        m_episode_length = 0;
        m_forward_reward = 0;
        m_ctrl_cost = 0;
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
    float get_x_velocity() const { return m_x_velocity; }
    float get_forward_reward() const { return m_forward_reward; }
    float get_ctrl_cost() const { return m_ctrl_cost; }
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

    float m_dt = 0.01f;
    int m_frame_skip = 5;
    float m_forward_reward_weight;
    float m_ctrl_cost_weight;

    int m_obs_dim = 0;
    int m_act_dim = 0;
    std::vector<float> m_obs_buffer;

    int m_episode_length = 0;
    float m_forward_reward = 0;
    float m_ctrl_cost = 0;
    float m_x_velocity = 0;

    std::mt19937 m_rng{42};
};

} // namespace joltgym

void bind_env(py::module_& m) {
    py::class_<joltgym::HalfCheetahCore>(m, "HalfCheetahCore")
        .def(py::init<const std::string&, float, float>(),
             py::arg("model_path"),
             py::arg("forward_reward_weight") = 1.0f,
             py::arg("ctrl_cost_weight") = 0.1f)
        .def("step", &joltgym::HalfCheetahCore::step)
        .def("reset", &joltgym::HalfCheetahCore::reset,
             py::arg("seed") = py::none(),
             py::arg("noise_scale") = 0.1f)
        .def("get_root_x", &joltgym::HalfCheetahCore::get_root_x)
        .def("get_x_velocity", &joltgym::HalfCheetahCore::get_x_velocity)
        .def("get_forward_reward", &joltgym::HalfCheetahCore::get_forward_reward)
        .def("get_ctrl_cost", &joltgym::HalfCheetahCore::get_ctrl_cost)
        .def("get_obs_dim", &joltgym::HalfCheetahCore::get_obs_dim)
        .def("get_action_dim", &joltgym::HalfCheetahCore::get_action_dim)
        .def("get_episode_length", &joltgym::HalfCheetahCore::get_episode_length)
        .def("shutdown", &joltgym::HalfCheetahCore::shutdown);
}
