
#include "matmul.cuh"
#include "matmul_kernels.cuh"

#include <iostream>

static constexpr dim3 kBlock(16, 16);

int main(int argc, char **argv)
{
	if (argc != 4) {
		std::cerr << "usage: matmul_test <M> <N> <K>\n";
		return EXIT_FAILURE;
	}
	const size_t m = std::stoul(argv[1]);
	const size_t n = std::stoul(argv[2]);
	const size_t k = std::stoul(argv[3]);

	MatMulWrapper<float> A(m, n, kRandomInit);
	MatMulWrapper<float> B(n, k, kRandomInit);
	MatMulWrapper<float> C(m, k, kEmptyInit);

	auto Cref = A.matrix * B.matrix;

	A.upload();
	B.upload();

	dim3 grid((C.cols + kBlock.x - 1) / kBlock.x, (A.rows + kBlock.y - 1) / kBlock.y);

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start));

	Kernels::matMulNaive<<<grid, kBlock>>>(A.dMatrix, B.dMatrix, C.dMatrix, A.rows, A.cols, B.cols);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float ms = 0;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
	std::printf("Kernel time: %.3f ms\n", ms);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	C.download();
}
