cmake_minimum_required(VERSION 3.20)
project(JoltGym VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(JOLTGYM_BUILD_RENDERER "Build Vulkan renderer (requires Vulkan SDK + glslc)" ON)
option(JOLTGYM_BUILD_PYTHON   "Build Python bindings" ON)
option(JOLTGYM_BUILD_TESTS    "Build C++ tests" OFF)

# Auto-detect Vulkan renderer dependencies
if(JOLTGYM_BUILD_RENDERER)
    find_package(Vulkan QUIET)
    find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin)
    if(NOT Vulkan_FOUND OR NOT GLSLC)
        if(NOT Vulkan_FOUND)
            message(STATUS "Vulkan not found — building without renderer (headless only)")
        else()
            message(STATUS "glslc not found — building without renderer (headless only)")
        endif()
        set(JOLTGYM_BUILD_RENDERER OFF CACHE BOOL "" FORCE)
    endif()
endif()

# Dependencies via FetchContent
include(cmake/FetchJolt.cmake)
include(cmake/FetchTinyXML2.cmake)

if(JOLTGYM_BUILD_RENDERER)
    include(cmake/FetchSDL2.cmake)
    include(cmake/FetchImGui.cmake)
    include(cmake/FetchVkBootstrap.cmake)
    include(cmake/FetchVMA.cmake)
endif()

if(JOLTGYM_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.13.6
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# Sub-libraries
add_subdirectory(src/core)
add_subdirectory(src/mjcf)

if(JOLTGYM_BUILD_RENDERER)
    add_subdirectory(src/shaders)
    add_subdirectory(src/renderer)
endif()

if(JOLTGYM_BUILD_PYTHON)
    add_subdirectory(src/bindings)
endif()

if(JOLTGYM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
