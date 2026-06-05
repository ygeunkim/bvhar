file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

if(NOT CPP_SOURCES)
    message(FATAL_ERROR "No C++ source files found in tests/")
endif()

# Catch2
include(../../cmake/utils.cmake)

find_or_fetch_package(
    Catch2
    3.5.3
    https://github.com/catchorg/Catch2.git
    "v3.5.3"
)

include(CTest)
enable_testing()

foreach(CPP_FILE ${CPP_SOURCES})
    file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CPP_FILE})
    get_filename_component(test ${CPP_FILE} NAME_WE)

    message(STATUS "Building tests:")
    message(STATUS "    Source: ${REL_PATH}")

    add_executable(${test} ${CPP_FILE})
    
    target_link_libraries(${test} PRIVATE
        bvhar::bvhar
        Catch2::Catch2WithMain
    )
    
    add_test(NAME ${test} COMMAND ${test})
endforeach()