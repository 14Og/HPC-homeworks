#include <cuda_runtime.h>

static constexpr size_t kMaxStencilSize{111};
extern __constant__ float kernel[kMaxStencilSize * kMaxStencilSize];
namespace Kernels {

__global__ void convolve2D(const uint8_t *aSrc, uint8_t *aDst, size_t aRows, size_t aCols,
	uint8_t aChannels, size_t aStencilSize)
{
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	size_t col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= aRows || col >= aCols)
		return;

	int half = static_cast<int>(aStencilSize) / 2;
	float weightSum{0};

	for (uint8_t ch = 0; ch < aChannels; ++ch) {
		float sum{0};

		for (int kr = -half; kr <= half; ++kr) {
			for (int kc = -half; kc <= half; ++kc) {
				// clamp neighbour coords to image bounds
				int nr  = static_cast<int>(row) + kr;
				int nc  = static_cast<int>(col) + kc;
				nr      = max(0, min(nr, static_cast<int>(aRows) - 1));
				nc      = max(0, min(nc, static_cast<int>(aCols) - 1));

				float w = kernel[(kr + half) * aStencilSize + (kc + half)];
				sum += aSrc[(nr * aCols + nc) * aChannels + ch] * w;

				if (ch == 0) // accumulate weight only once
					weightSum += w;
			}
		}

		aDst[(row * aCols + col) * aChannels + ch]
			= static_cast<uint8_t>(fminf(fmaxf(sum / weightSum, 0.f), 255.f));
	}
}
}; // namespace Kernels