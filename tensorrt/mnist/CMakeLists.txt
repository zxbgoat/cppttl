# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/usr/local/tensorrt/include)
link_directories(/usr/local/tensorrt/lib)

include_directories(${PROJECT_SOURCE_DIR}/common)
link_directories(${PROJECT_SOURCE_DIR}/lib)

# add_executable(mnist ../common/logger.cpp mnist.cpp)
add_executable(mnist mnist.cpp)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

target_link_libraries(mnist
                     nvinfer
                     nvonnxparser
                     nvparsers
                     cudart
                     logger)