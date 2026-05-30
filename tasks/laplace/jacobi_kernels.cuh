#ifndef TASKS_LAPLACE_JACOBI_KERNELS_CUH_
#define TASKS_LAPLACE_JACOBI_KERNELS_CUH_
#include <cuda_runtime.h>

namespace Kernels {

__global__ void jacobi(const float *aCurr, float *aNext, int aN)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= aN * aN)
		return;

	size_t iy = id / aN;
	size_t ix = id % aN;

	if (iy == 0 || iy == aN - 1 || ix == 0 || ix == aN - 1)
		return;

	// clang-format off
	aNext[id] = 0.25f
		* (aCurr[(iy - 1) * aN + ix] 
        + aCurr[(iy + 1) * aN + ix] 
        + aCurr[id - 1] 
        + aCurr[id + 1]);
	// clang-format on
}

__global__ void blockMaxDiff(const float *aCurr, const float *aNext, float *blockMax, int aN)
{
	extern __shared__ float sdata[];

	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	auto tx   = threadIdx.x;
	size_t iy = id / aN;
	size_t ix = id % aN;
	if (id >= aN * aN || iy == 0 || iy == aN - 1 || ix == 0 || ix == aN - 1)
		sdata[tx] = 0.0f;
	else
		sdata[tx] = fabsf(aCurr[id] - aNext[id]);

	__syncthreads();

	// tree reduction
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tx < s)
			sdata[tx] = fmaxf(sdata[tx], sdata[tx + s]);
		__syncthreads();
	}
	if (tx == 0)
		blockMax[blockIdx.x] = sdata[0];
}

}; // namespace Kernels

#endif /* TASKS_LAPLACE_JACOBI_KERNELS_CUH_ */
