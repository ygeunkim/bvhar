file(GLOB_RECURSE CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

if(NOT CPP_SOURCES)
    message(FATAL_ERROR "No C++ source files found in examples/")
endif()

foreach(CPP_FILE ${CPP_SOURCES})
    file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CPP_FILE})
    get_filename_component(example ${CPP_FILE} NAME_WE)

    message(STATUS "Building examples:")
    message(STATUS "    Source: ${REL_PATH}")

    add_executable(${example} ${CPP_FILE})
    set_property(TARGET ${example} PROPERTY CXX_STANDARD 17)
    
    target_link_libraries(${example} PRIVATE bvhar)
    # target_link_libraries(${example} PRIVATE bvhar::bvhar)
endforeach()