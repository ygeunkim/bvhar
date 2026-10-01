include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)

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
