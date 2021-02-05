# tensorrt
include_directories(/usr/local/tensorrt/include
                    /usr/local/tensorrt/samples/common)
link_directories(/usr/local/tensorrt/lib)

add_executable(mnist mnist.cpp)
