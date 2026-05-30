#ifndef TASKS_HISTOGRAM_HISTOGRAM_KERNELS_CUH_
#define TASKS_HISTOGRAM_HISTOGRAM_KERNELS_CUH_

#include <cstdint>
#include <cuda_runtime.h>

namespace Kernels {

__global__ void BGR2Grayscale(const uint8_t *aSrc, uint8_t *aDst, size_t aRows, size_t aCols)
{
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	size_t col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= aRows || col >= aCols)
		return;

	size_t grayOffset = row * aCols + col;
	size_t bgrOffset  = grayOffset * 3;

	auto b            = aSrc[bgrOffset];
	auto g            = aSrc[bgrOffset + 1];
	auto r            = aSrc[bgrOffset + 2];

    auto gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
	aDst[grayOffset]  = gray > 255 ? 255 : gray;
}

__global__ void histogram(const uint8_t *aImg, size_t aRows, size_t aCols, int *aGlobalHist)
{
	__shared__ int localHist[256];

	for (size_t i = threadIdx.y * blockDim.x + threadIdx.x; i < 256; i += blockDim.x * blockDim.y)
		localHist[i] = 0;
	__syncthreads();

	size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	size_t col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < aRows && col < aCols) {
		auto id = row * aCols + col;
		atomicAdd(&localHist[aImg[id]], 1);
	}
	__syncthreads();

	for (size_t i = threadIdx.y * blockDim.x + threadIdx.x; i < 256; i += blockDim.x * blockDim.y)
		atomicAdd(&aGlobalHist[i], localHist[i]);
}

}; // namespace Kernels

#endif /* TASKS_HISTOGRAM_HISTOGRAM_KERNELS_CUH_ */
