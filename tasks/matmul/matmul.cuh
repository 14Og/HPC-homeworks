#ifndef TASKS_MATMUL_MATMUL_CUH_
#define TASKS_MATMUL_MATMUL_CUH_

#include "Eigen/Dense"
#include <stdexcept>

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

    float *matrixData()
    {
        if (!matrix)
            throw std::runtime_error("MatMulWrapper::matrixData(): empty host matrix");
        return matrix.data();
    }
    
    float *deviceMatrixData()
    {
        if (!dMatrix)
            throw std::runtime_error("MatMulWrapper::deviceMatrixData(): empty host matrix");
        return dMatrix;
    }

private:
    size_t rows{0};
    size_t cols{0};
	MatT matrix;
	float *dMatrix{nullptr};
};

#endif /* TASKS_MATMUL_MATMUL_CUH_ */
