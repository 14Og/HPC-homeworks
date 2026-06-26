#ifndef TASKS_MATMUL_MATMUL_CUH_
#define TASKS_MATMUL_MATMUL_CUH_

#include "Eigen/Dense"

template<typename T>
class MatMulWrapper {

    using MatT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::RowMajor>;

    MatMulWrapper(size_t aRows, size_t aCols): rows(aRows), cols(aCols), matrix(MatT::Random(aRows, aCols))
    {   
        CUDA_CHECK(cudaMalloc(&dMatrix, numBytes()));
    }

	MatMulWrapper(const MatMulWrapper &)            = delete;
	MatMulWrapper &operator=(const MatMulWrapper &) = delete;
	MatMulWrapper(MatMulWrapper &&)                 = delete;
	MatMulWrapper &operator=(MatMulWrapper &&)      = delete;

	void upload() // Upload host matrix to device
	{
	}

	void download() // Download device matrix to host
	{

	}

	size_t numBytes()
	{
		return rows * cols * sizeof(T);
	}

private:
    size_t rows{0};
    size_t cols{0};
	MatT matrix;
	float *dMatrix{nullptr};
};

#endif /* TASKS_MATMUL_MATMUL_CUH_ */
