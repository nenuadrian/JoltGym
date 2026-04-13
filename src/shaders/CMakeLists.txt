# GLSLC is already found and validated by the root CMakeLists.txt
# before JOLTGYM_BUILD_RENDERER is enabled.

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SPIRV_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

function(compile_shader SHADER_FILE)
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
    string(REPLACE "." "_" SHADER_VAR ${SHADER_NAME})
    set(SPIRV_FILE ${SPIRV_OUTPUT_DIR}/${SHADER_NAME}.spv)
    set(HEADER_FILE ${SPIRV_OUTPUT_DIR}/${SHADER_VAR}.h)

    add_custom_command(
        OUTPUT ${SPIRV_FILE}
        COMMAND ${GLSLC} ${SHADER_DIR}/${SHADER_FILE} -o ${SPIRV_FILE}
        DEPENDS ${SHADER_DIR}/${SHADER_FILE}
        COMMENT "Compiling ${SHADER_FILE} to SPIR-V"
    )

    add_custom_command(
        OUTPUT ${HEADER_FILE}
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_spirv.cmake
            ${SPIRV_FILE} ${HEADER_FILE} ${SHADER_VAR}
        DEPENDS ${SPIRV_FILE}
        COMMENT "Embedding ${SHADER_NAME}.spv as C header"
    )

    set(SHADER_HEADERS ${SHADER_HEADERS} ${HEADER_FILE} PARENT_SCOPE)
endfunction()

compile_shader(phong.vert)
compile_shader(phong.frag)

add_custom_target(compile_shaders ALL DEPENDS ${SHADER_HEADERS})

# Make the generated headers accessible
add_library(joltgym_shaders INTERFACE)
target_include_directories(joltgym_shaders INTERFACE ${SPIRV_OUTPUT_DIR})
add_dependencies(joltgym_shaders compile_shaders)
