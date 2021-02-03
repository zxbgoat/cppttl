cmake_minimum_required(VERSION 3.17)
project(mnist)

set(CMAKE_CXX_STANDARD 11)

# cuda
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAG} -gencode arch=compute_61,code=[sm_61,compute_61]")
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
aux_source_directory(/usr/local/tensorrt/samples/common COMMON)
include_directories(/usr/local/tensorrt/include
        /usr/local/tensorrt/samples/common)
link_directories(/usr/local/tensorrt/lib)

add_executable(mnist
        mnist.cpp)
