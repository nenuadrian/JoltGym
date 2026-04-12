add_library(joltgym_core STATIC
    joltgym_core.cpp
    physics_world.cpp
    body_registry.cpp
    articulation.cpp
    motor_controller.cpp
    state_extractor.cpp
    collision_layers.cpp
)

target_include_directories(joltgym_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(joltgym_core PUBLIC Jolt)
