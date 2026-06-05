#ifndef TASKS_MATMUL_MATMUL_KERNELS_CUH_
#define TASKS_MATMUL_MATMUL_KERNELS_CUH_

#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

namespace Kernels {

__global__ void matMulNaive(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= k)
        return;
    
    float value = 0;
    for (size_t i = 0; i < n; ++i) {
        value += A[row * n + i] * B[i * k + col];
    }

    C[row * k + col] = value;
}

}; // namespace Kernels

#endif /* TASKS_MATMUL_MATMUL_KERNELS_CUH_ */
