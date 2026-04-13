/// @file renderer.h
/// @brief Abstract renderer interface for Vulkan-based visualization.
#pragma once

#include <cstdint>
#include <vector>
#include <array>

namespace joltgym {

/// @brief Supported primitive geometry types for rendering.
enum class PrimitiveType { Capsule, Sphere, Box, Cylinder, Plane };

/// @brief Transform and appearance data for rendering a single primitive.
struct RenderTransform {
    float model[16];      ///< 4x4 column-major model matrix.
    float color[4];       ///< RGBA color.
    PrimitiveType type;   ///< Primitive geometry type.
    float scale[3];       ///< Additional scale factors.
};

/// @brief Abstract base class for renderers.
///
/// Concrete implementations include SwapchainRenderer (SDL2 window + ImGui)
/// and OffscreenRenderer (headless framebuffer for `rgb_array` mode).
class Renderer {
public:
    virtual ~Renderer() = default;

    /// @brief Initialize the renderer with the given viewport size.
    virtual bool Init(uint32_t width, uint32_t height) = 0;

    /// @brief Begin a new frame.
    virtual void BeginFrame() = 0;

    /// @brief Draw a single primitive with the given transform.
    virtual void DrawPrimitive(const RenderTransform& transform) = 0;

    /// @brief Finish the frame and present/store the result.
    virtual void EndFrame() = 0;

    /// @brief Shut down the renderer and release GPU resources.
    virtual void Shutdown() = 0;

    /// @brief Whether the window has been closed (SwapchainRenderer only).
    virtual bool ShouldClose() const { return false; }

    /// @brief Process window events (SwapchainRenderer only).
    virtual void PollEvents() {}

    /// @brief Retrieve the rendered frame as RGBA pixel data.
    /// @param[out] width  Image width in pixels.
    /// @param[out] height Image height in pixels.
    /// @return RGBA bytes (row-major, 4 bytes per pixel).
    virtual std::vector<uint8_t> GetFrameBuffer(uint32_t& width, uint32_t& height) {
        width = height = 0;
        return {};
    }

    /// @brief Set the view matrix (4x4 column-major).
    virtual void SetViewMatrix(const float* view_4x4) = 0;

    /// @brief Set the projection matrix (4x4 column-major).
    virtual void SetProjectionMatrix(const float* proj_4x4) = 0;

    /// @brief Set the directional light direction.
    virtual void SetLightDirection(float x, float y, float z) = 0;

    /// @brief Set the camera world position (for specular lighting).
    virtual void SetCameraPosition(float x, float y, float z) = 0;
};

} // namespace joltgym
