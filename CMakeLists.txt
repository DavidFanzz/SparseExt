cmake_minimum_required(VERSION 3.1)

macro(set_torch_flags)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    if(Torch_VERSION VERSION_LESS "1.6")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} TORCH_V1_5")
    endif()
endmacro()

if ($ENV{TRAVIS_BUILD_NUMBER})
    set(PROJECT_VERSION 0.0.$ENV{TRAVIS_BUILD_NUMBER})
else ()
    set(PROJECT_VERSION 0.0.1)
endif ()

PROJECT(sparse_turbo VERSION ${PROJECT_VERSION} LANGUAGES C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp -march=native")
find_package(Threads)
find_package(Torch REQUIRED)
set_torch_flags()
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

add_subdirectory(third_party/pybind11)

add_subdirectory(src)
