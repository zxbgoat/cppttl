include_directories(/usr/local/tensorrt/include)

add_library(logger SHARED logger.cpp)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)