set(TASKS laplace filter histogram)
set(OPENCV_TASKS filter histogram)

add_compile_definitions(
    OUT_DIR="${PROJECT_SOURCE_DIR}/assets"
    MEDIA_DIR="${PROJECT_SOURCE_DIR}/media"
    BLUR_DIR="${PROJECT_SOURCE_DIR}/assets/blur"
    HIST_DIR="${PROJECT_SOURCE_DIR}/assets/hist"
    GRAY_DIR="${PROJECT_SOURCE_DIR}/assets/gray"
)

foreach(TASK ${TASKS})
    add_executable(${TASK} ${TASK}/${TASK}.cu)
    set_target_properties(${TASK} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${TASK} PRIVATE ${CMAKE_SOURCE_DIR}/tasks)
endforeach()

foreach(TASK ${OPENCV_TASKS})
    target_link_libraries(${TASK} PRIVATE ${OPENCV_LIBS})
endforeach()