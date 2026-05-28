add_executable(Laplace laplace.cu)
set_target_properties(Laplace PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(Laplace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
