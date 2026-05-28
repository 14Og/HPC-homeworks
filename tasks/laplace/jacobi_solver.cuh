#ifndef TASKS_LAPLACE_JACOBI_SOLVER_HH_
#define TASKS_LAPLACE_JACOBI_SOLVER_HH_

#include <algorithm>
#include <cstdio>
#include <vector>
#include <cmath>

#include "check.hh"
#include "grid.hh"
#include "jacobi_kernels.cuh"

class JacobiSolver {
public:
	JacobiSolver(float aTol, int aMaxIter, int blockSize = 256) : tol(aTol), maxIter(aMaxIter), blockSize(blockSize)
	{
	}

	// Returns the number of iterations performed.
	// Populates g with the converged solution (current device buffer).
	int solve(Grid &g)
	{
		const auto total     = g.size() * g.size();
		const auto numBlocks = (total + blockSize - 1) / blockSize;

		float *dBlockMaxes   = nullptr;
		CUDA_CHECK(cudaMalloc(&dBlockMaxes, numBlocks * sizeof(float)));

		std::vector<float> hBlockMaxes(numBlocks);
		dim3 dimGrid(numBlocks);
		dim3 dimBlock(blockSize);
		int iter = 0;

		for (; iter < maxIter; ++iter) {
			Kernels::jacobi<<<dimGrid, dimBlock>>>(g.devCurr(), g.devNext(), g.size());

			CUDA_CHECK(cudaMemset(dBlockMaxes, 0, numBlocks * sizeof(float)));
			Kernels::blockMaxDiff<<<dimGrid, dimBlock, blockSize * sizeof(float)>>>(
				g.devCurr(), g.devNext(), dBlockMaxes, g.size());
			g.swapBuffers();

			CUDA_CHECK(cudaMemcpy(hBlockMaxes.data(), dBlockMaxes, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
			auto maxDiff = *std::max_element(hBlockMaxes.begin(), hBlockMaxes.end());
			if (maxDiff < tol) {
				++iter;
				break;
			}
		}
		CUDA_CHECK(cudaFree(dBlockMaxes));

		return iter;
	}

private:
	float tol{0.0};
	int maxIter{0};
	int blockSize{0};
};

#endif /* TASKS_LAPLACE_JACOBI_SOLVER_HH_ */
