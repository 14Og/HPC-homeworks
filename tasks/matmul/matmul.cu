#include "matmul.cuh"
#include "matmul_kernels.cuh"

static constexpr dim3 kBlock(16, 16);

int main()
{
    MatMulWrapper<float> A(100, 100, kRandomInit);
    MatMulWrapper<float> B(100, 100, kRandomInit);
    MatMulWrapper<float> C(100, 100, kEmptyInit);

    A.upload();
    B.upload();

    Kernels::matMulNaive(A.matrix.data(), B.matrix.data(), C.dMatrix, A.rows, A.cols, B.cols);

}