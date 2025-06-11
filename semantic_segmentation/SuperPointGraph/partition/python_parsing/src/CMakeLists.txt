cmake_minimum_required(VERSION 3.1)

get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME})

##############################
### Find required packages ###
##############################
find_package(PythonLibs)
find_package(PythonInterp)
include(FindNumPy)
include_directories(../include)

include_directories(${PROJECT_NAME} PRIVATE ${SegmentParsingSPG_INCLUDE_DIR})
include_directories(${PROJECT_NAME} PRIVATE ${SegmentParsingSPG_glew_INCLUDE_DIR})
include_directories(${PROJECT_NAME} PRIVATE ${SegmentParsingSPG_glfw_INCLUDE_DIR})
include_directories(${PROJECT_NAME} PRIVATE GLEW_STATIC)


find_package(Boost 1.69.0 COMPONENTS graph REQUIRED) #system filesystem thread serialization
if (${Boost_MINOR_VERSION} LESS 67 )
    find_package(Boost 1.65.0 COMPONENTS numpy${PYTHON_VERSION_MAJOR} REQUIRED) #system filesystem thread serialization
else()
    set(PYTHONVERSION ${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
    find_package(Boost 1.69.0 COMPONENTS numpy REQUIRED)
endif()
include_directories(C:/dev/boost_numpy/Boost.NumPy)
link_directories(${YOUR_DIRECTORY})
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})



message("PYTHON LIBRARIES ARE " ${PYTHON_LIBRARIES})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})
LINK_DIRECTORIES(${PYTHON_LIBRARY_DIRS})

file(GLOB python_parsing_HEADERS include/*.h)

set(CMAKE_LD_FLAG "${CMAKE_LD_FLAGS} -shared -Wl -fPIC --export-dynamic -o -O3 -Wall -std=c++11")
message(${Boost_LIBRARIES})
add_library(python_parsing SHARED python_parsing.cpp ${python_parsing_HEADERS})
target_link_libraries(python_parsing ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} easy3d)