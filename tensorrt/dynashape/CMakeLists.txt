# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/usr/local/tensorrt/include
                    /usr/local/tensorrt/samples/common)
link_directories(/usr/local/tensorrt/lib)

add_executable(dynashape dynashape.cpp)