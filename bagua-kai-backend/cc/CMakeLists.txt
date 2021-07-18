cmake_minimum_required(VERSION 3.14)
project(bagua_kai_backend)

# GoogleTest requires at least C++11
set(CMAKE_CXX_STANDARD 11)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    bagua_kai_backend
    bagua_kai_backend.cc
)
target_link_libraries(
    bagua_kai_backend
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(bagua_kai_backend)
