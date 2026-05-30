#ifndef TASKS_CHECK_CUH_
#define TASKS_CHECK_CUH_

#include <cstdio>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        std::exit(1); } } while(0)


#endif /* TASKS_CHECK_CUH_ */
