cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Get version number for DESCRIPTION
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/DESCRIPTION" DESC_CONTENT)
string(REGEX MATCH "Version:[ \t]*([0-9.]+)" _ "${DESC_CONTENT}")
if(CMAKE_MATCH_1)
    set(BVHAR_VERSION "${CMAKE_MATCH_1}")
    message(STATUS "Detected bvhar version from DESCRIPTION: ${BVHAR_VERSION}")
else()
    message(WARNING "Could not parse version from DESCRIPTION file. Defaulting to 0.0.0.9000")
    set(BVHAR_VERSION "0.0.0.9000")
endif()

project(bvhar VERSION ${BVHAR_VERSION} LANGUAGES CXX)

# Required libraries
find_package(Eigen3 REQUIRED)
message(STATUS "Found Eigen3")
find_package(Boost REQUIRED)
message(STATUS "Found Boost")
# find_package(lbfgspp REQUIRED)
find_package(spdlog REQUIRED)
message(STATUS "Found spdlog")
# Optional OpenMP
find_package(OpenMP QUIET)

# Manually find lbfgs
find_path(LBFGSPP_INCLUDE_DIR NAMES LBFGS.h PATH_SUFFIXES lbfgspp)
if(NOT LBFGSPP_INCLUDE_DIR)
    message(FATAL_ERROR "lbfgspp headers not found.")
endif()
add_library(lbfgspp INTERFACE IMPORTED)
target_include_directories(lbfgspp INTERFACE ${LBFGSPP_INCLUDE_DIR})
message(STATUS "Found lbfgspp")

# Header-only library
add_library(bvhar INTERFACE)
add_library(bvhar::bvhar ALIAS bvhar)

target_compile_definitions(bvhar INTERFACE
    EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
    BOOST_ENABLE_ASSERT_HANDLER
    FMT_HEADER_ONLY
    BVHAR_VERSION="${PROJECT_VERSION}"
)

target_include_directories(bvhar INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inst/include>
)

target_link_libraries(bvhar INTERFACE
    Eigen3::Eigen
    spdlog::spdlog
    lbfgspp
)

if(TARGET Boost::boost)
    target_link_libraries(bvhar INTERFACE Boost::boost)
endif()

# Boost::boost does not work on linux and windows
if(TARGET Boost::random)
    target_link_libraries(bvhar INTERFACE
        Boost::random
        Boost::math
        Boost::accumulators
        Boost::optional
    )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(bvhar INTERFACE OpenMP::OpenMP_CXX)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(BVHAR_BUILD_EXAMPLES "Build example C++ programs" OFF)

if(BVHAR_BUILD_EXAMPLES)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    
    add_subdirectory(inst/examples examples)
endif()
