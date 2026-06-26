#ifndef TASKS_MATMUL_MATMUL_CUH_
#define TASKS_MATMUL_MATMUL_CUH_

#include "Eigen/Dense"

template<typename T>
struct MatMulWrapper {

	Eigen::Matrix<T, Eigen::Dynamic, Eigen::RowMajor> matrix;
	float *dMatrix{nullptr};
	size_t rows{0};
	size_t cols{0};

	void upload()
	{
	}

	void download()
	{
	}

	size_t numBytes()
	{
		return 0;
	}
};

#endif /* TASKS_MATMUL_MATMUL_CUH_ */
