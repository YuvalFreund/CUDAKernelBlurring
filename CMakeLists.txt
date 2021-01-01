project(filtering)
cmake_minimum_required(VERSION 3.1)

# Set the default build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

find_package(CUDA REQUIRED)

# # Fermi
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=sm_20")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=sm_21")

# Kepler
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

# # Maxwell
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

cuda_add_executable(gauss_filter
	common.cc
	main.cc
	filtering.cc
	image.cc
	conv_gpu.cu
	conv_cpu.cc)

# Activate (and require) C++11 support
set_property(TARGET gauss_filter PROPERTY CXX_STANDARD 11)
set_property(TARGET gauss_filter PROPERTY CXX_STANDARD_REQUIRED ON)