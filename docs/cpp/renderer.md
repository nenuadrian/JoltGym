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

::: doxy.joltgym-cpp.Class.joltgym::Renderer

---

## RenderTransform

::: doxy.joltgym-cpp.Class.joltgym::RenderTransform

---

## PrimitiveType

::: doxy.joltgym-cpp.Enum.joltgym::PrimitiveType
