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

# External libraries
include(FetchContent)

function(find_or_fetch_package PackageName version GIT_REPOSITORY GIT_TAG)
    find_package(${PackageName} ${version} QUIET)
    if(NOT ${PackageName}_FOUND)
        message(STATUS "Could not find ${PackageName}, fetching from ${GIT_REPOSITORY}")
        FetchContent_Declare(
            ${PackageName}
            GIT_REPOSITORY ${GIT_REPOSITORY}
            GIT_TAG ${GIT_TAG}
            GIT_SHALLOW TRUE
            OVERRIDE_FIND_PACKAGE
        )
        # FetchContent_MakeAvailable(${PackageName})
        # lbfgspp uses find_package(Eigen3 3.0 REQUIRED)
        if(${PackageName} STREQUAL "lbfgspp")
            FetchContent_GetProperties(lbfgspp)
            if(NOT lbfgspp_POPULATED)
                # FetchContent_Populate is deprecated in the recent cmake
                cmake_policy(PUSH)
                if(POLICY CMP0169)
                    cmake_policy(SET CMP0169 OLD)
                endif()
                FetchContent_Populate(${PackageName})
                cmake_policy(POP)

                add_library(lbfgspp INTERFACE)
                target_include_directories(lbfgspp INTERFACE ${lbfgspp_SOURCE_DIR}/include)
                target_link_libraries(lbfgspp INTERFACE Eigen3::Eigen)
            endif()
        else()
            FetchContent_MakeAvailable(${PackageName})
        endif()
    else()
        message(STATUS "Found ${PackageName}")
    endif()
endfunction()

# atomic linking issue in llvm (macos, windows): dummy atomic
if ((APPLE OR WIN32) AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_library(atomic INTERFACE)
endif()

# Specify boost components before fetching
set(BOOST_INCLUDE_LIBRARIES random math accumulators optional)

# Required libraries
find_or_fetch_package(
    Eigen3
    3.4
    https://gitlab.com/libeigen/eigen.git
    "3.4.1"
)
find_or_fetch_package(
    Boost
    1.87.0
    https://github.com/boostorg/boost.git
    "boost-1.87.0"
)
find_or_fetch_package(
    lbfgspp
    0.4.0
    https://github.com/yixuan/LBFGSpp.git
    "v0.4.0"
)
find_or_fetch_package(
    spdlog
    1.15.3
    https://github.com/gabime/spdlog.git
    "v1.15.3"
)
# Optional OpenMP
find_package(OpenMP QUIET)

# Header-only library
add_library(bvhar INTERFACE)
add_library(bvhar::bvhar ALIAS bvhar)

target_compile_features(bvhar INTERFACE cxx_std_17)

# Add /bigobj for MSVC to handle large object files
target_compile_options(bvhar INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
)

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

# Boost::boost does not work
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
option(BVHAR_BUILD_TESTS "Build test C++ programs" OFF)
option(BVHAR_ENABLE_COVERAGE "Enable C++ test coverage" OFF)

if(BVHAR_BUILD_EXAMPLES)
    add_subdirectory(inst/examples examples)
endif()

if(BVHAR_BUILD_TESTS)
    add_subdirectory(inst/tests tests)
endif()

if(BVHAR_ENABLE_COVERAGE)
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()
