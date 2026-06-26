#ifndef TASKS_MATMUL_MATMUL_CUH_
#define TASKS_MATMUL_MATMUL_CUH_

#include "Eigen/Dense"

#include "check.cuh"


struct RandomInit {};
struct EmptyInit {};
static constexpr auto kRandomInit = RandomInit();
static constexpr auto kEmptyInit  = EmptyInit();

template<typename T>
class MatMulWrapper {

	using MatT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::RowMajor>;

	MatMulWrapper(size_t aRows, size_t aCols) : rows(aRows), cols(aCols)
	{
		CUDA_CHECK(cudaMalloc(&dMatrix, numBytes()));
	}

public:
	MatMulWrapper(size_t aRows, size_t aCols, RandomInit) : MatMulWrapper(aRows, aCols)
	{
		matrix = MatT::Random(rows, cols);
	}

	MatMulWrapper(size_t aRows, size_t aCols, EmptyInit) : MatMulWrapper(aRows, aCols)
	{
		matrix.resize(rows, cols);
	}

	MatMulWrapper(const MatMulWrapper &)            = delete;
	MatMulWrapper &operator=(const MatMulWrapper &) = delete;
	MatMulWrapper(MatMulWrapper &&)                 = delete;
	MatMulWrapper &operator=(MatMulWrapper &&)      = delete;

	void upload() // Upload host matrix to device
	{
		CUDA_CHECK(cudaMemcpy(dMatrix, matrix.data(), numBytes(), cudaMemcpyHostToDevice));
	}

	void download() // Download device matrix to host
	{
		CUDA_CHECK(cudaMemcpy(matrix.data(), dMatrix, cudaMemcpyDeviceToHost));
	}

	size_t numBytes()
	{
		return rows * cols * sizeof(T);
	}

public:
	size_t rows{0};
	size_t cols{0};
	MatT matrix;
	float *dMatrix{nullptr};
};

#endif /* TASKS_MATMUL_MATMUL_CUH_ */
