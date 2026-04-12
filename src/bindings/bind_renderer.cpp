#include <pybind11/pybind11.h>

// Renderer bindings — placeholder for now
// Full bindings will be added when renderer is stable

namespace py = pybind11;

void bind_renderer(py::module_& m) {
    // Will expose SwapchainRenderer and OffscreenRenderer
    // For now, rendering is handled through the HalfCheetahCore class
}
