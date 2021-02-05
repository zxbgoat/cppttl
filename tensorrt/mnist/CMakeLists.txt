# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/usr/local/tensorrt/include)
link_directories(/usr/local/tensorrt/lib)

include_directories(${PROJECT_SOURCE_DIR}/common)
link_directories(${PROJECT_SOURCE_DIR}/common)

add_executable(mnist mnist.cpp)

target_link_libraries(mnist
                     nvinfer
                     nvonnxparser
                     cudart
                     logger)