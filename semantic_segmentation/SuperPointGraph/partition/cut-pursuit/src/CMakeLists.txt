
include_directories(../include)

##############################
### Find required packages ###
##############################
find_package(Boost COMPONENTS graph numpy37 REQUIRED)

set(conda_env_path "/home/dgeo/anaconda3/envs/spg")
list(APPEND CMAKE_PREFIX_PATH ${conda_env_path})

find_package(PythonLibs)
find_package(PythonInterp)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
find_package(NumPy REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})

message("Boost INCLUDE DIRS ARE " ${Boost_INCLUDE_DIRS})
message("Boost LIBRARIES DIRS ARE " ${Boost_LIBRARY_DIRS})
message("Boost LIBRARIES ARE " ${Boost_LIBRARIES})
message("Python LIBRARIES ARE " ${PYTHON_LIBRARIES})
message("Python INCLUDE DIR ARE " ${PYTHON_NUMPY_INCLUDE_DIR})
message("Python-Numpy INCLUDE DIR ARE " ${PYTHON_NUMPY_INCLUDE_DIR})

file(GLOB CP_HEADERS include/*.h)

set(CMAKE_LD_FLAG "${CMAKE_LD_FLAGS} -shared -Wl -fPIC --export-dynamic -o -O3 -Wall -std=c++11")

add_library(cp SHARED cutpursuit.cpp ${CP_HEADERS})
target_link_libraries(cp
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES})