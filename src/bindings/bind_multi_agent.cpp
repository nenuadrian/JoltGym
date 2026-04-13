#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "multi_agent_env.h"

namespace py = pybind11;

void bind_multi_agent(py::module_& m) {
    py::class_<joltgym::MultiAgentEnv>(m, "MultiAgentEnv")
        .def(py::init<int, const std::string&, float, float, float>(),
             py::arg("num_agents"),
             py::arg("model_path"),
             py::arg("agent_spacing") = 3.0f,
             py::arg("forward_reward_weight") = 1.0f,
             py::arg("ctrl_cost_weight") = 0.1f)
        .def_property_readonly("num_agents", &joltgym::MultiAgentEnv::num_agents)
        .def_property_readonly("obs_dim", &joltgym::MultiAgentEnv::obs_dim)
        .def_property_readonly("act_dim", &joltgym::MultiAgentEnv::act_dim)
        .def("step", [](joltgym::MultiAgentEnv& env, py::array_t<float> actions) {
            int n = env.num_agents();
            int obs_dim = env.obs_dim();

            auto obs = py::array_t<float>({n, obs_dim});
            auto rewards = py::array_t<float>(n);

            {
                py::gil_scoped_release release;
                env.Step(actions.data(), obs.mutable_data(), rewards.mutable_data());
            }

            return py::make_tuple(obs, rewards);
        }, py::arg("actions"))
        .def("reset_all", [](joltgym::MultiAgentEnv& env,
                             std::optional<uint32_t> seed, float noise_scale) {
            int n = env.num_agents();
            int obs_dim = env.obs_dim();
            auto obs = py::array_t<float>({n, obs_dim});

            {
                py::gil_scoped_release release;
                env.ResetAll(obs.mutable_data(), seed, noise_scale);
            }

            return obs;
        }, py::arg("seed") = py::none(), py::arg("noise_scale") = 0.1f)
        .def("get_agent_x", &joltgym::MultiAgentEnv::GetAgentX, py::arg("idx"))
        .def("get_agent_x_velocity", &joltgym::MultiAgentEnv::GetAgentXVelocity, py::arg("idx"));
}
