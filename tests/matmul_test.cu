#include <cstdlib>
#include <iostream>

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "check.cuh"
#include "matmul/matmul_kernels.cuh"

static constexpr float kTol = 1e-3f;
static constexpr dim3 kBlock(16, 16);

int main(int argc, char **argv)
{
	if (argc != 4) {
		std::cerr << "usage: matmul_test <M> <N> <K>\n";
		return EXIT_FAILURE;
	}

	const size_t M = std::stoul(argv[1]);
	const size_t N = std::stoul(argv[2]);
	const size_t K = std::stoul(argv[3]);

	using RowMatXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

	RowMatXf A     = RowMatXf::Random(M, N);
	RowMatXf B     = RowMatXf::Random(N, K);
	RowMatXf Cref  = A * B;

	float *dA, *dB, *dC;
	CUDA_CHECK(cudaMalloc(&dA, M * N * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&dB, N * K * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&dC, M * K * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(dA, A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dB, B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));

	dim3 grid((K + kBlock.x - 1) / kBlock.x, (M + kBlock.y - 1) / kBlock.y);
	Kernels::matMulNaive<<<grid, kBlock>>>(dA, dB, dC, M, N, K);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	RowMatXf C(M, K);
	CUDA_CHECK(cudaMemcpy(C.data(), dC, M * K * sizeof(float), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dA));
	CUDA_CHECK(cudaFree(dB));
	CUDA_CHECK(cudaFree(dC));

	float maxErr = (C - Cref).cwiseAbs().maxCoeff();
	if (maxErr > kTol) {
		std::cerr << "FAIL: max error " << maxErr << " > " << kTol << "\n";
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
