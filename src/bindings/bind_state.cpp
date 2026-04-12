#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "core/state_extractor.h"

namespace py = pybind11;

void bind_state(py::module_& m) {
    py::class_<joltgym::StateExtractor>(m, "StateExtractor")
        .def("get_obs_dim", &joltgym::StateExtractor::GetObsDim)
        .def("get_qpos_dim", &joltgym::StateExtractor::GetQPosDim)
        .def("get_qvel_dim", &joltgym::StateExtractor::GetQVelDim)
        .def("get_action_dim", &joltgym::StateExtractor::GetActionDim)
        .def("get_root_x", &joltgym::StateExtractor::GetRootX)
        .def("get_root_x_velocity", &joltgym::StateExtractor::GetRootXVelocity);
}
