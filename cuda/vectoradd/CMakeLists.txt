project(vectoradd LANGUAGES CXX CUDA)

link_directories(include)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib)

add_executable(vectoradd vectoradd.cu)
