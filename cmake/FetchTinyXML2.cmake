include(FetchContent)

FetchContent_Declare(
    tinyxml2
    GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
    GIT_TAG        10.0.0
)

set(tinyxml2_BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(tinyxml2)
