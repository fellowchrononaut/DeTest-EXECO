# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MeshMatching)

# Set the C++ standard to 17 for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Point Cloud Library (PCL) package and its components.
# This command is still necessary to find the header files.
find_package(PCL 1.10 REQUIRED COMPONENTS 
    common 
    io 
    features 
    keypoints 
    registration 
    visualization
    filters 
)

# Include the PCL directories so the compiler can find the header files
include_directories(${PCL_INCLUDE_DIRS})

# Fetch and configure yaml-cpp using FetchContent
include(FetchContent)
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.7.0 # Use a specific tag for stability
)
FetchContent_MakeAvailable(yaml-cpp)

# Add the executable to be built from our source file
add_executable(mesh_matching mesh_matching.cpp)

# Link the executable against the required libraries.
# FINAL ATTEMPT: We are now manually listing every required PCL library.
# This is the most explicit method and bypasses any issues with CMake variables.
target_link_libraries(mesh_matching 
    pcl_common
    pcl_io
    pcl_features
    pcl_keypoints
    pcl_registration
    pcl_visualization
    pcl_filters  # <-- The crucial library that was not being linked
    yaml-cpp
    lz4
)
target_compile_definitions(mesh_matching PRIVATE PCL_NO_PRECOMPILE)