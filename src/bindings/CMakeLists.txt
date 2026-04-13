pybind11_add_module(joltgym_native MODULE
    module.cpp
    bind_world.cpp
    bind_renderer.cpp
    bind_state.cpp
    bind_env.cpp
    world_pool.cpp
    multi_agent_env.cpp
    bind_multi_agent.cpp
)

target_include_directories(joltgym_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(joltgym_native PRIVATE
    joltgym_core
    joltgym_mjcf
)

if(JOLTGYM_BUILD_RENDERER)
    target_link_libraries(joltgym_native PRIVATE joltgym_renderer)
    target_compile_definitions(joltgym_native PRIVATE JOLTGYM_HAS_RENDERER=1)
endif()

install(TARGETS joltgym_native DESTINATION joltgym COMPONENT python)
