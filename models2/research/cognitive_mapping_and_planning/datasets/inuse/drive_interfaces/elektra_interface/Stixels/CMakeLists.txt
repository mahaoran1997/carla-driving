#    This file is part of stixels. (https://github.com/dhernandez0/stixels).
#
#    Copyright (c) 2016 Daniel Hernandez Juarez.

#    stixels is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    stixels is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with stixels.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.1.0)
project(stixels)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")
SET(VERSION_REQ_OCV "3.1")
SET(VERSION_REQ_CUDA "8.0")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
IF(WIN32) #Windows
	if(CMAKE_CL_64)
		SET(ZED_INCLUDE_DIRS $ENV{ZED_INCLUDE_DIRS})
		SET(ZED_LIBRARIES $ENV{ZED_LIBRARIES_64})
		SET(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})
		SET(OPENCV_DIR $ENV{OPENCV_DIR})
	else()
		message(FATAL_ERROR "You've selected the 32bit version of ${CMAKE_GENERATOR}. \n Please delete the cache (file->Delete Cache) and use the 64bit version. (${CMAKE_GENERATOR} Win64)")
    endif()
ELSE() #Linux
	find_package(ZED 2.0 REQUIRED)
	
	##FOR Jetson, OpenCV4Tegra is based on OpenCV2.4
	exec_program(uname ARGS -p OUTPUT_VARIABLE CMAKE_SYSTEM_NAME2)
	if (CMAKE_SYSTEM_NAME2 MATCHES "aarch64")
		SET(VERSION_REQ_OCV "2.4")
		SET(VERSION_REQ_CUDA "")

		unset(CUDA_USE_STATIC_CUDA_RUNTIME CACHE)
		option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
		message("CUDA_USE_STATIC_CUDA_RUNTIME : ${CUDA_USE_STATIC_CUDA_RUNTIME}")
	endif()
ENDIF(WIN32)

find_package(OpenCV ${VERSION_REQ_OCV} REQUIRED)
find_package(CUDA ${VERSION_REQ_CUDA} REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 --use_fast_math
    -gencode=arch=compute_30,code=sm_62
    -gencode=arch=compute_30,code=sm_62
    -gencode=arch=compute_30,code=sm_62
    -gencode=arch=compute_30,code=sm_62
    )


link_directories(${ZED_LIBRARY_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
	


cuda_add_executable(
    stixels main.cu
    RoadEstimation.cu RoadEstimationKernels.cu Stixels.cu StixelsKernels.cu
    median_filter.cu hamming_cost.cu disparity_method.cu debug.cu costs.cu)
#add_executable(stixels1 main.cpp)
#CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS(~/ stixels_final stixels stixels1)
#add_definitions(-std=c++0x -g -03)
#add_definitions (-std=gnu++11)
#
#nvcc -std=c++11 main.cu 
#target_compile_options(stixels PRIVATE cxx_range_for)

target_link_libraries( stixels 
			${ZED_LIBRARIES}
                        ${OpenCV_LIBRARIES}
			${CUDA_CUDA_LIBRARY}
 			${CUDA_CUDART_LIBRARY}
			${CUDA_npp_LIBRARY}
)

