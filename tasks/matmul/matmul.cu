#include "matmul.cuh"
#include "matmul_kernels.cuh"

static constexpr dim3 kBlock(16, 16);

int main()
{
	MatMulWrapper<float> A(100, 100, kRandomInit);
	MatMulWrapper<float> B(100, 100, kRandomInit);
	MatMulWrapper<float> C(100, 100, kEmptyInit);

	auto Cref = A.matrix * B.matrix;

	A.upload();
	B.upload();

	dim3 grid((C.cols + kBlock.x - 1) / kBlock.x, (A.rows + kBlock.y - 1) / kBlock.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
	Kernels::matMulNaive<<<kBlock, grid>>>(
		A.matrix.data(), B.matrix.data(), C.dMatrix, A.rows, A.cols, B.cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    

    C.download();
    
}
