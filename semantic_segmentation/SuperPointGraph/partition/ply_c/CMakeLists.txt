# Graph for Cut Pursuit
# author: Loic Landrieu
# date: 2017

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(LIBGEOF)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -fopenmp -O3")

##############################
### Find required packages ###
##############################
find_package(Boost COMPONENTS graph numpy37 REQUIRED)

set (conda_env_path "/home/dgeo/anaconda3/envs/spg")
list(APPEND CMAKE_PREFIX_PATH ${conda_env_path})

find_package(PythonLibs)
find_package(PythonInterp)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
find_package(NumPy REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})

find_package(Eigen3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

message("Boost INCLUDE DIRS ARE " ${Boost_INCLUDE_DIRS})
message("Boost LIBRARIES ARE " ${Boost_LIBRARY_DIRS})
message("Python LIBRARIES ARE " ${PYTHON_LIBRARIES})
message("Python INCLUDE DIR ARE " ${PYTHON_NUMPY_INCLUDE_DIR})
message("Python-Numpy INCLUDE DIR ARE " ${PYTHON_NUMPY_INCLUDE_DIR})
message("EIGEN3 INCLUDE DIR ARE " ${EIGEN3_INCLUDE_DIR})

##############################
###  Build target library  ###
##############################

set(CMAKE_LD_FLAG "${CMAKE_LD_FLAGS} -shared -Wl -fPIC --export-dynamic -fopenmp -O3 -Wall")

add_library(ply_c SHARED ply_c.cpp)
target_link_libraries(ply_c
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES}
    )
