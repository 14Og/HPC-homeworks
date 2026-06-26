add_executable(matmul_test matmul_test.cu)
set_target_properties(matmul_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(matmul_test PRIVATE ${CMAKE_SOURCE_DIR}/tasks)
target_link_libraries(matmul_test PRIVATE Eigen3::Eigen)

# M:N:K
set(MATMUL_TEST_CASES
    "1:1:1"           
    "1:256:1"         
    "16:16:16"        
    "128:256:64"      
)

foreach(CASE ${MATMUL_TEST_CASES})
    string(REPLACE ":" "x" CASE_NAME "${CASE}")
    string(REPLACE ":" ";" CASE_ARGS "${CASE}")
    add_test(NAME "matmul_naive_${CASE_NAME}" COMMAND matmul_test ${CASE_ARGS})
endforeach()
