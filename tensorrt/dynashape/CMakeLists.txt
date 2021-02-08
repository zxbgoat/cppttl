include_directories(/usr/local/tensorrt/include
                    /usr/local/tensorrt/samples/common
                    /usr/local/cuda/include)
link_directories(/usr/local/tensorrt/lib
                 /usr/local/cuda/lib64)

add_executable(dynashape dynashape.cpp)