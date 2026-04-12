#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_world(py::module_& m);
void bind_renderer(py::module_& m);
void bind_state(py::module_& m);
void bind_env(py::module_& m);

PYBIND11_MODULE(joltgym_native, m) {
    m.doc() = "JoltGym native bindings — MuJoCo-compatible physics simulation for RL";

    bind_world(m);
    bind_state(m);
    bind_env(m);

    #ifdef JOLTGYM_HAS_RENDERER
    bind_renderer(m);
    #endif
}
