set(TASKS laplace filter)
set(OPENCV_TASKS filter)

add_compile_definitions(
    OUT_DIR="${PROJECT_SOURCE_DIR}/assets"
    MEDIA_DIR="${PROJECT_SOURCE_DIR}/media"
)



foreach(TASK ${TASKS})
    add_executable(${TASK} ${TASK}/${TASK}.cu)
    set_target_properties(${TASK} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${TASK} PRIVATE ${CMAKE_SOURCE_DIR}/tasks)
endforeach()

foreach(TASK ${OPENCV_TASKS})
    target_link_libraries(${TASK} PRIVATE ${OPENCV_LIBS})
endforeach()