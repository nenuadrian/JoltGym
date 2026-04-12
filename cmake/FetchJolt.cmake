include(FetchContent)

FetchContent_Declare(
    JoltPhysics
    GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
    GIT_TAG        v5.2.0
    SOURCE_SUBDIR  Build
)

# Enable deterministic simulation across platforms (critical for RL reproducibility)
set(CROSS_PLATFORM_DETERMINISTIC ON CACHE BOOL "" FORCE)
set(INTERPROCEDURAL_OPTIMIZATION ON CACHE BOOL "" FORCE)
set(FLOATING_POINT_EXCEPTIONS_ENABLED OFF CACHE BOOL "" FORCE)

# Fix for Apple Clang overriding fp-model conflict
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    add_compile_options(-Wno-overriding-option)
endif()
# Disable Jolt's own test/sample builds
set(TARGET_HELLO_WORLD OFF CACHE BOOL "" FORCE)
set(TARGET_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(TARGET_PERFORMANCE_TEST OFF CACHE BOOL "" FORCE)
set(TARGET_SAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(JoltPhysics)
