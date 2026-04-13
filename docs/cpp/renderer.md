# Renderer

JoltGym includes an optional Vulkan-based renderer with SDL2 windowing and Dear ImGui debug overlays. Build controlled by the `JOLTGYM_BUILD_RENDERER` CMake option (default: `ON`).

```
Renderer (abstract interface)
  |-- SwapchainRenderer  (SDL2 window + ImGui overlay)
  +-- OffscreenRenderer  (headless framebuffer -> RGBA bytes)
       |
       +-- VulkanContext (device, instance, queues)
            |-- Pipeline (vertex/fragment shaders)
            |-- MeshPrimitives (capsule, sphere, box, cylinder)
            +-- Camera (view/projection matrices)
```

When disabled, the core physics library builds without any Vulkan/SDL2 dependencies, suitable for headless training servers:

```bash
cmake -B build -DJOLTGYM_BUILD_RENDERER=OFF
```

## Renderer Interface

Abstract base class for renderers. Concrete implementations include `SwapchainRenderer` (SDL2 window + ImGui) and `OffscreenRenderer` (headless framebuffer for `rgb_array` mode).

```cpp
class Renderer {
public:
    virtual ~Renderer() = default;

    virtual bool Init(uint32_t width, uint32_t height) = 0;
    virtual void BeginFrame() = 0;
    virtual void DrawPrimitive(const RenderTransform& transform) = 0;
    virtual void EndFrame() = 0;
    virtual void Shutdown() = 0;

    virtual bool ShouldClose() const;  // Window closed (SwapchainRenderer)
    virtual void PollEvents();         // Process window events

    /// Retrieve rendered frame as RGBA pixel data (row-major, 4 bytes/pixel).
    virtual std::vector<uint8_t> GetFrameBuffer(uint32_t& width, uint32_t& height);

    virtual void SetViewMatrix(const float* view_4x4) = 0;
    virtual void SetProjectionMatrix(const float* proj_4x4) = 0;
    virtual void SetLightDirection(float x, float y, float z) = 0;
    virtual void SetCameraPosition(float x, float y, float z) = 0;
};
```

---

## RenderTransform

Transform and appearance data for rendering a single primitive.

```cpp
struct RenderTransform {
    float model[16];      // 4x4 column-major model matrix
    float color[4];       // RGBA color
    PrimitiveType type;   // Primitive geometry type
    float scale[3];       // Additional scale factors
};
```

---

## PrimitiveType

Supported primitive geometry types for rendering.

```cpp
enum class PrimitiveType { Capsule, Sphere, Box, Cylinder, Plane };
```
