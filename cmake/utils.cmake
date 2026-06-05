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