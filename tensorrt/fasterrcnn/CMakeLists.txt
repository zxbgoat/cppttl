cmake_minimum_required(VERSION 3.10)

project(fasterrcnn)

set(WORKDIR ~/Workspace/cppttl/tensorrt/)
set(CUDADIR /usr/local/cuda)
set(SOURCES
        src/fasterrcnn.cpp)

add_executable(fasterrcnn ${SOURCES})

target_include_directories(fasterrcnn
        PRIVATE
            ${WORKDIR}/common
            ${CUDADIR}/include
            include
        )