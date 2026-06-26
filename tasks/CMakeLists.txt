set(TASKS matmul)

foreach(TASK ${TASKS})
    add_executable(${TASK} ${TASK}/${TASK}.cu)
    set_target_properties(${TASK} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${TASK} PRIVATE ${CMAKE_SOURCE_DIR}/tasks)
    target_link_libraries(${TASK} PRIVATE Eigen3::Eigen)
endforeach()

