include(FetchContent)

FetchContent_Declare(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG        v1.3.296
)

set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(vk-bootstrap)
