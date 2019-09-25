cmake_minimum_required(VERSION 2.8)
project( extract_frames )
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -std=c++14 -Wall -Wextra")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Ofast -flto")

find_package( OpenCV REQUIRED )
#find_package( Boost COMPONENTS filesystem system thread program_options serialization iostreams REQUIRED )

#include_directories( ${Boost_INCLUDE_DIR} )

set (CUDA_PROPAGATE_HOST_FLAGS off)

add_executable( extract_frames extract_frames.cpp)
target_link_libraries( extract_frames ${OpenCV_LIBS} )
