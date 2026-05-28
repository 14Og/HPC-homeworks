#ifndef TASKS_LAPLACE_JACOBI_SOLVER_HH_
#define TASKS_LAPLACE_JACOBI_SOLVER_HH_

#include <algorithm>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "../check.hh"
#include "grid.hh"

// ---------------------------------------------------------------------------
// Kernel declarations (defined in laplace.cu)
// ---------------------------------------------------------------------------

namespace Kernels {

// One Jacobi sweep: next[iy,ix] = 0.25 * (curr[iy-1,ix] + curr[iy+1,ix]
//                                        + curr[iy,ix-1] + curr[iy,ix+1])
// Only interior points are updated; boundary cells are left untouched.
__global__ void jacobi(const float *aCurr, float *aNext, int aNx, int aNy);

// Per-block max of |curr[i] - next[i]| over interior points.
// Result for each block written to block_maxes[blockIdx.x].
// Launch with shared memory: block_size * sizeof(float)
__global__ void blockMaxDiff(const float *aCurr, const float *aNext, float *blockMax, int aNx, int aNy);

}; // namespace Kernels

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

class JacobiSolver {
public:
	JacobiSolver(float aTol, int aMaxIter, int blockSize = 256) :
		tol(aTol), maxIter(aMaxIter), blockSize(blockSize)
	{
	}

	// Returns the number of iterations performed.
	// Populates g with the converged solution (current device buffer).
	int solve(Grid &g)
	{
		const int total     = g.nX * g.nY;
		const int numBlocks = (total + blockSize - 1) / blockSize;

		// TODO: allocate d_block_maxes (numBlocks floats) with cudaMalloc
		float *dBlockMaxes = nullptr;

		std::vector<float> hBlockMaxes(numBlocks);

		int iter = 0;
		for (; iter < maxIter; ++iter) {

			// TODO: launch Kernels::jacobi
			//   grid  : dim3(numBlocks, 1)   threads: dim3(blockSize, 1)
			//   args  : g.devCurr(), g.devNext(), g.nX, g.nY

			// TODO: reset dBlockMaxes to 0 before reduction
			//   hint: cudaMemset

			// TODO: launch Kernels::blockMaxDiff
			//   grid  : dim3(numBlocks, 1)   threads: dim3(blockSize, 1)
			//   shared: blockSize * sizeof(float)
			//   args  : g.devCurr(), g.devNext(), dBlockMaxes, g.nX, g.nY

			g.swapBuffers();

			// Reduce block maxima on the CPU (numBlocks is small)
			// TODO: cudaMemcpy d_block_maxes -> hBlockMaxes.data()  (DeviceToHost)
			const float max_diff = *std::max_element(hBlockMaxes.begin(), hBlockMaxes.end());
			if (max_diff < tol) {
				++iter;
				break;
			}
		}

		// TODO: cudaFree d_block_maxes

		return iter;
	}

private:
	float tol{0.0};
	int maxIter{0};
	int blockSize{0};
};

#endif /* TASKS_LAPLACE_JACOBI_SOLVER_HH_ */
