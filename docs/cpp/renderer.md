# Renderer

JoltGym includes an optional Vulkan-based renderer with SDL2 windowing and Dear ImGui debug overlays. The renderer supports both on-screen (interactive) and off-screen (headless) modes.

## Architecture

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

## Renderer Interface

`src/renderer/renderer.h`

The abstract base class for all renderers:

```cpp
class Renderer {
public:
    virtual ~Renderer() = default;

    virtual bool Init(uint32_t width, uint32_t height) = 0;
    virtual void BeginFrame() = 0;
    virtual void DrawPrimitive(const RenderTransform& transform) = 0;
    virtual void EndFrame() = 0;
    virtual void Shutdown() = 0;

    virtual bool ShouldClose() const;
    virtual void PollEvents();

    // Off-screen: retrieve rendered image as RGBA bytes
    virtual std::vector<uint8_t> GetFrameBuffer(
        uint32_t& width, uint32_t& height);

    // Camera and lighting
    virtual void SetViewMatrix(const float* view_4x4) = 0;
    virtual void SetProjectionMatrix(const float* proj_4x4) = 0;
    virtual void SetLightDirection(float x, float y, float z) = 0;
    virtual void SetCameraPosition(float x, float y, float z) = 0;
};
```

## Render Primitives

Each physics shape is rendered as a primitive with a transform:

```cpp
enum class PrimitiveType { Capsule, Sphere, Box, Cylinder, Plane };

struct RenderTransform {
    float model[16];      // 4x4 column-major model matrix
    float color[4];       // RGBA
    PrimitiveType type;
    float scale[3];       // additional scale factors
};
```

## Components

### VulkanContext

`src/renderer/vulkan_context.h`

Manages Vulkan initialization using vk-bootstrap:

- Instance and device creation
- Queue selection
- Memory allocation (VMA)
- Command buffer management

### SwapchainRenderer

`src/renderer/swapchain_renderer.h`

Window-based rendering with:

- SDL2 window management
- Vulkan swapchain presentation
- Dear ImGui integration for debug overlays
- Camera controls

### OffscreenRenderer

`src/renderer/offscreen_renderer.h`

Headless rendering for `rgb_array` mode:

- Renders to an off-screen framebuffer
- Returns pixel data as `std::vector<uint8_t>`
- No window or display required

### SceneSync

`src/renderer/scene_sync.h`

Synchronizes physics state to render geometry:

- Reads body positions/rotations from Jolt
- Generates `RenderTransform` for each geom
- Called each frame before rendering

### Camera

`src/renderer/camera.h`

Orbital camera with:

- Position, target, up vector
- Perspective projection
- Mouse-based rotation and zoom (SwapchainRenderer only)

### MeshPrimitives

`src/renderer/mesh_primitives.h`

Generates vertex/index buffers for each primitive type (capsule, sphere, box, cylinder).

### ImGui Layer

`src/renderer/imgui_layer.h`

Debug overlay showing:

- Joint positions and velocities
- Motor torques
- Physics step timing
- Body positions

## Build Configuration

The renderer is controlled by the `JOLTGYM_BUILD_RENDERER` CMake option (default: `ON`).

When disabled, the core physics library builds without any Vulkan/SDL2 dependencies, making it suitable for headless training servers.

```bash
# Build without renderer
cmake -B build -DJOLTGYM_BUILD_RENDERER=OFF
```
