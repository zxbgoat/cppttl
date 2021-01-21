project(HELLOLIB)
add_subdirectory(lib)
get_target_property(OUTPUT_VALUE hello_static OUTPUT_NAME)
message(STATUS "This is he hello_static OUTPUT_NAME:" ${OUTPUT_VALUE})