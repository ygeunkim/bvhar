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

enable_testing()

foreach(CPP_FILE ${CPP_SOURCES})
    file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CPP_FILE})
    get_filename_component(test ${CPP_FILE} NAME_WE)

    message(STATUS "Building tests:")
    message(STATUS "    Source: ${REL_PATH}")

    add_executable(${test} ${CPP_FILE})
    # set_property(TARGET ${test} PROPERTY CXX_STANDARD 17)
    
    target_link_libraries(${test} PRIVATE bvhar::bvhar)
    target_link_libraries(${test} PRIVATE Catch2::Catch2WithMain)
    
    add_test(NAME ${test} COMMAND ${test} --order decl)
    set_tests_properties(${test} PROPERTIES RUN_SERIAL ON)
endforeach()