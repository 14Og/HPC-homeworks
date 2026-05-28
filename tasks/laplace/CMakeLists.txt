set(TARGET_NAME laplace)
add_executable(${TARGET_NAME} laplace.cu)
set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/tasks)
target_compile_definitions(${TARGET_NAME} PRIVATE OUT_DIR="${PROJECT_SOURCE_DIR}/assets")
