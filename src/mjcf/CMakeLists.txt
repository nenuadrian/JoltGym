add_library(joltgym_mjcf STATIC
    mjcf_model.cpp
    mjcf_defaults.cpp
    mjcf_compiler.cpp
    mjcf_parser.cpp
    mjcf_to_jolt.cpp
)

target_include_directories(joltgym_mjcf PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(joltgym_mjcf PUBLIC joltgym_core tinyxml2)
