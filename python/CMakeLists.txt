cmake_minimum_required(VERSION 3.10)
project(bvhar)

# Pybind11
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Prioritize conda
if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}" ${CMAKE_PREFIX_PATH})
    set(CMAKE_INCLUDE_PATH "$ENV{CONDA_PREFIX}/include" ${CMAKE_INCLUDE_PATH})
    set(CMAKE_LIBRARY_PATH "$ENV{CONDA_PREFIX}/lib" ${CMAKE_LIBRARY_PATH})
    set(CMAKE_FIND_ROOT_PATH "$ENV{CONDA_PREFIX}")
    # set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    # set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    message(STATUS "Using conda environment: $ENV{CONDA_PREFIX}")
endif()

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
        )
        FetchContent_MakeAvailable(${PackageName})
        if(${PackageName} STREQUAL "lbfgspp" AND NOT TARGET LBFGSpp::LBFGSpp)
            add_library(LBFGSpp INTERFACE)
            target_include_directories(LBFGSpp INTERFACE ${${PackageName}_SOURCE_DIR}/include)
            add_library(LBFGSpp::LBFGSpp ALIAS LBFGSpp)
        endif()
    else()
        message(STATUS "Found ${PackageName}")
    endif()
endfunction()

# Eigen
find_or_fetch_package(
    Eigen3
    3.4
    https://gitlab.com/libeigen/eigen.git
    "3.4.1"
)
# boost
find_or_fetch_package(
    Boost
    1.88.0
    https://github.com/boostorg/boost.git
    "boost-1.88.0"
)
# LBFGSpp
find_or_fetch_package(
    lbfgspp
    0.4.0
    https://github.com/yixuan/LBFGSpp.git
    "v0.4.0"
)
# spdlog
find_or_fetch_package(
    spdlog
    1.15.3
    https://github.com/gabime/spdlog.git
    "v1.15.3"
)

find_package(OpenMP QUIET)

# Pybind11 sources
file(GLOB_RECURSE CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cpp")

if(NOT CPP_SOURCES)
    message(FATAL_ERROR "No C++ source files found in src/")
endif()

# execute_process(
#     COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('purelib'))"
#     OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )

# pybind11_add_module(bvhar ${CPP_SOURCES})
foreach(CPP_FILE ${CPP_SOURCES})
    file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src ${CPP_FILE})
    get_filename_component(MODULE_DIR ${REL_PATH} DIRECTORY)
    get_filename_component(MODULE_NAME ${CPP_FILE} NAME_WE)

    if(MODULE_DIR)
        string(REPLACE "/" "." MODULE_PATH ${MODULE_DIR})
        set(FULL_MODULE_NAME "${MODULE_PATH}.${MODULE_NAME}")
        set(INSTALL_DIR "${MODULE_DIR}")
    else()
        set(FULL_MODULE_NAME ${MODULE_NAME})
        set(INSTALL_DIR ".")
    endif()

    pybind11_add_module(${FULL_MODULE_NAME} ${CPP_FILE})
    
    target_include_directories(${FULL_MODULE_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../inst/include
    )

    # target_link_libraries(${FULL_MODULE_NAME} PRIVATE
    #     Eigen3::Eigen
    #     Boost::boost
    #     spdlog::spdlog
    #     LBFGSpp::LBFGSpp
    # )
    if(TARGET Eigen3::Eigen)
        target_link_libraries(${FULL_MODULE_NAME} PRIVATE Eigen3::Eigen)
    endif()

    if(TARGET Boost::boost)
        target_link_libraries(${FULL_MODULE_NAME} PRIVATE Boost::boost)
    endif()

    if(TARGET spdlog::spdlog)
        target_link_libraries(${FULL_MODULE_NAME} PRIVATE spdlog::spdlog)
    endif()

    if(TARGET LBFGSpp::LBFGSpp)
        target_link_libraries(${FULL_MODULE_NAME} PRIVATE LBFGSpp::LBFGSpp)
    endif()

    target_compile_definitions(${FULL_MODULE_NAME} PRIVATE
        EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
        BOOST_ENABLE_ASSERT_HANDLER
        FMT_HEADER_ONLY
    )

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${FULL_MODULE_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    install(TARGETS ${FULL_MODULE_NAME}
        LIBRARY DESTINATION ${MODULE_DIR}
        RUNTIME DESTINATION ${MODULE_DIR}
    )
endforeach()

# Install Python source files
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/bvhar
#     DESTINATION .
#     FILES_MATCHING 
#     PATTERN "*.py"
#     PATTERN "__pycache__" EXCLUDE
# )
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
    DESTINATION "${SKBUILD_HEADERS_DIR}"
)

# bvhar header
# # include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../inst/include)
# target_include_directories(bvhar PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../inst/include>
#     $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
# )
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../inst/include/
    DESTINATION include
)
