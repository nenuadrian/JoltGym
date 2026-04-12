#pragma once

#include <cstdint>
#include <vector>
#include <array>

namespace joltgym {

enum class PrimitiveType { Capsule, Sphere, Box, Cylinder, Plane };

struct RenderTransform {
    float model[16];      // 4x4 column-major model matrix
    float color[4];       // RGBA
    PrimitiveType type;
    float scale[3];       // Additional scale factors
};

class Renderer {
public:
    virtual ~Renderer() = default;
    virtual bool Init(uint32_t width, uint32_t height) = 0;
    virtual void BeginFrame() = 0;
    virtual void DrawPrimitive(const RenderTransform& transform) = 0;
    virtual void EndFrame() = 0;
    virtual void Shutdown() = 0;

    virtual bool ShouldClose() const { return false; }
    virtual void PollEvents() {}

    // For offscreen: get the rendered image as RGBA bytes
    virtual std::vector<uint8_t> GetFrameBuffer(uint32_t& width, uint32_t& height) {
        width = height = 0;
        return {};
    }

    virtual void SetViewMatrix(const float* view_4x4) = 0;
    virtual void SetProjectionMatrix(const float* proj_4x4) = 0;
    virtual void SetLightDirection(float x, float y, float z) = 0;
    virtual void SetCameraPosition(float x, float y, float z) = 0;
};

} // namespace joltgym
