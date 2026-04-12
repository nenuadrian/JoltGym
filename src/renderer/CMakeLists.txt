add_library(joltgym_renderer STATIC
    renderer.cpp
    vulkan_context.cpp
    swapchain_renderer.cpp
    offscreen_renderer.cpp
    imgui_layer.cpp
    pipeline.cpp
    mesh_primitives.cpp
    camera.cpp
    scene_sync.cpp
)

target_include_directories(joltgym_renderer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(joltgym_renderer PUBLIC
    joltgym_core
    Vulkan::Vulkan
    SDL2::SDL2-static
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    imgui
    joltgym_shaders
)
