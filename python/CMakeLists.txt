cmake_minimum_required(VERSION 3.10...4.0)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(bvhar)

# Pybind11
# find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
# find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
# find_package(pybind11 REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Prioritize conda
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Using conda environment: $ENV{CONDA_PREFIX}")
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}" ${CMAKE_PREFIX_PATH})
    set(CMAKE_INCLUDE_PATH "$ENV{CONDA_PREFIX}/include" ${CMAKE_INCLUDE_PATH})
    set(CMAKE_LIBRARY_PATH "$ENV{CONDA_PREFIX}/lib" ${CMAKE_LIBRARY_PATH})
    set(CMAKE_FIND_ROOT_PATH "$ENV{CONDA_PREFIX}")
    set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
    set(CMAKE_INSTALL_INCLUDEDIR "$ENV{CONDA_PREFIX}/include")
    # set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    # set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(BOOST_ROOT "$ENV{CONDA_PREFIX}")
    set(BOOST_INCLUDEDIR "$ENV{CONDA_PREFIX}/include")
    set(BOOST_LIBRARYDIR "$ENV{CONDA_PREFIX}/lib")
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

# Eigen
find_or_fetch_package(
    Eigen3
    3.4
    https://gitlab.com/libeigen/eigen.git
    "3.4.1"
)
# boost
cmake_policy(PUSH)
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()
find_or_fetch_package(
    Boost
    1.84.0
    https://github.com/boostorg/boost.git
    "boost-1.88.0"
)
cmake_policy(POP)
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

add_library(bvhar INTERFACE)

target_include_directories(bvhar INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../inst/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# bvhar header in site-packages/bvhar/include
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../inst/include/
    DESTINATION include
)
# bvhar header in include
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../inst/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

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

foreach(CPP_FILE ${CPP_SOURCES})
    file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src ${CPP_FILE})
    get_filename_component(MODULE_DIR ${REL_PATH} DIRECTORY)
    get_filename_component(MODULE_NAME ${CPP_FILE} NAME_WE)

    string(REPLACE "/" "." MODULE_PATH ${MODULE_DIR})
    set(FULL_MODULE_NAME "${MODULE_PATH}.${MODULE_NAME}")
    set(INSTALL_DIR "${MODULE_DIR}")

    # if(MODULE_DIR MATCHES "^bvhar/(.*)$")
    #     set(MODULE_DIR "${CMAKE_MATCH_1}")
    # elseif(MODULE_DIR STREQUAL "bvhar")
    #     set(MODULE_DIR "")
    # endif()

    # if(MODULE_DIR)
    #     string(REPLACE "/" "." MODULE_PATH ${MODULE_DIR})
    #     set(FULL_MODULE_NAME "bvhar.${MODULE_PATH}.${MODULE_NAME}")
    #     set(INSTALL_DIR "bvhar/${MODULE_DIR}")
    # else()
    #     set(FULL_MODULE_NAME "bvhar.${MODULE_NAME}")
    #     set(INSTALL_DIR "bvhar")
    # endif()

    string(REPLACE "/" "_" TARGET_NAME "${MODULE_DIR}_${MODULE_NAME}")
    string(REPLACE "." "_" TARGET_NAME "${TARGET_NAME}")

    # message(STATUS "FULL_MODULE_NAME: ${FULL_MODULE_NAME}")
    # message(STATUS "MODULE_NAME: ${MODULE_NAME}")
    # message(STATUS "MODULE_DIR: ${MODULE_DIR}")
    # message(STATUS "INSTALL_DIR: ${INSTALL_DIR}")
    # message(STATUS "CPP_FILE: ${CPP_FILE}")
    # message(STATUS "REL_PATH: ${REL_PATH}")
    message(STATUS "Building module:")
    message(STATUS "    Source: ${REL_PATH}")
    message(STATUS "    Module name: ${FULL_MODULE_NAME}")
    message(STATUS "    Install dir: ${INSTALL_DIR}")

    # pybind11_add_module(${FULL_MODULE_NAME} ${CPP_FILE})
    pybind11_add_module(${TARGET_NAME} MODULE ${CPP_FILE})
    
    # target_include_directories(${FULL_MODULE_NAME} PRIVATE
    #     ${CMAKE_CURRENT_SOURCE_DIR}/../inst/include
    # )
    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${MODULE_NAME})
    
    target_link_libraries(${TARGET_NAME} PRIVATE bvhar)

    target_link_libraries(${TARGET_NAME} PRIVATE
        Eigen3::Eigen
        Boost::boost
        # Boost::random
        # Boost::math
        # Boost::accumulators
        # Boost::optional
        spdlog::spdlog
        lbfgspp
    )
    # if(TARGET Eigen3::Eigen)
    #     target_link_libraries(${TARGET_NAME} PRIVATE Eigen3::Eigen)
    # endif()

    # if(TARGET Boost::headers)
    #     target_link_libraries(${TARGET_NAME} PRIVATE Boost::headers)
    # endif()

    # if(TARGET spdlog::spdlog)
    #     target_link_libraries(${TARGET_NAME} PRIVATE spdlog::spdlog)
    # endif()

    # # if(TARGET LBFGSpp::LBFGSpp)
    # #     target_link_libraries(${FULL_MODULE_NAME} PRIVATE LBFGSpp::LBFGSpp)
    # # endif()
    # if(TARGET lbfgspp)
    #     target_link_libraries(${TARGET_NAME} PRIVATE lbfgspp)
    # endif()

    target_compile_definitions(${TARGET_NAME} PRIVATE
        EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
        BOOST_ENABLE_ASSERT_HANDLER
        FMT_HEADER_ONLY
    )

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # install(TARGETS ${FULL_MODULE_NAME}
    #     LIBRARY DESTINATION ${MODULE_DIR}
    #     RUNTIME DESTINATION ${MODULE_DIR}
    # )
    install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${INSTALL_DIR}
        COMPONENT python_modules
    )
endforeach()

# Install Python source files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/bvhar
    DESTINATION .
    FILES_MATCHING 
    PATTERN "*.py"
    PATTERN "*.csv"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.cpp" EXCLUDE
)
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
#     DESTINATION "${SKBUILD_HEADERS_DIR}"
# )

# # bvhar header in site-packages/bvhar/include
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../inst/include/
#     DESTINATION include
# )
# # bvhar header in include
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../inst/include/
#     DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )
