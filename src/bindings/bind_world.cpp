#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "core/physics_world.h"
#include "core/joltgym_core.h"
#include "world_pool.h"
#include "mjcf/mjcf_parser.h"
#include "mjcf/mjcf_to_jolt.h"

namespace py = pybind11;

void bind_world(py::module_& m) {
    py::class_<joltgym::PhysicsWorld>(m, "PhysicsWorld")
        .def(py::init<>())
        .def("init", &joltgym::PhysicsWorld::Init,
             py::arg("max_bodies") = 2048,
             py::arg("max_body_pairs") = 4096,
             py::arg("max_contact_constraints") = 2048,
             py::arg("single_threaded") = false)
        .def("step", &joltgym::PhysicsWorld::Step,
             py::arg("dt"), py::arg("collision_steps") = 1)
        .def("save_snapshot", &joltgym::PhysicsWorld::SaveSnapshot)
        .def("restore_snapshot", &joltgym::PhysicsWorld::RestoreSnapshot);

    py::class_<joltgym::MjcfParser>(m, "MjcfParser")
        .def(py::init<>())
        .def("parse", &joltgym::MjcfParser::Parse);

    py::class_<joltgym::WorldPool>(m, "WorldPool")
        .def(py::init<int, const std::string&, float, float>(),
             py::arg("num_envs"),
             py::arg("model_path"),
             py::arg("forward_reward_weight") = 1.0f,
             py::arg("ctrl_cost_weight") = 0.1f)
        .def_property_readonly("num_envs", &joltgym::WorldPool::num_envs)
        .def_property_readonly("obs_dim", &joltgym::WorldPool::obs_dim)
        .def_property_readonly("act_dim", &joltgym::WorldPool::act_dim)
        .def("step_all", [](joltgym::WorldPool& pool, py::array_t<float> actions) {
            int n = pool.num_envs();
            int obs_dim = pool.obs_dim();

            auto obs = py::array_t<float>({n, obs_dim});
            auto rewards = py::array_t<float>(n);
            auto dones_arr = py::array_t<bool>(n);

            {
                py::gil_scoped_release release;
                pool.StepAll(actions.data(), obs.mutable_data(),
                             rewards.mutable_data(),
                             reinterpret_cast<bool*>(dones_arr.mutable_data()));
            }

            return py::make_tuple(obs, rewards, dones_arr);
        }, py::arg("actions"))
        .def("reset_all", [](joltgym::WorldPool& pool,
                             std::optional<uint32_t> seed, float noise_scale) {
            int n = pool.num_envs();
            int obs_dim = pool.obs_dim();
            auto obs = py::array_t<float>({n, obs_dim});

            py::gil_scoped_release release;
            pool.ResetAll(obs.mutable_data(), seed, noise_scale);

            return obs;
        }, py::arg("seed") = py::none(), py::arg("noise_scale") = 0.1f)
        .def("reset_one", [](joltgym::WorldPool& pool, int idx,
                             std::optional<uint32_t> seed, float noise_scale) {
            int obs_dim = pool.obs_dim();
            auto obs = py::array_t<float>(obs_dim);

            py::gil_scoped_release release;
            pool.ResetOne(idx, obs.mutable_data(), seed, noise_scale);

            return obs;
        }, py::arg("idx"), py::arg("seed") = py::none(), py::arg("noise_scale") = 0.1f);

    m.def("init", &joltgym::JoltGymCore::Init);
    m.def("shutdown", &joltgym::JoltGymCore::Shutdown);
}
