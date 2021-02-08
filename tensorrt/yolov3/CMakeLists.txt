# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBS})

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/usr/local/tensorrt/include
                    /usr/local/tensorrt/samples/common)
link_directories(/usr/local/tensorrt/lib)

add_executable(yolov3 yolov3.cpp)
