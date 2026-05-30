#include <cuda_runtime.h>

static constexpr size_t kMaxStencilSize{20};
extern __constant__ float kernel[kMaxStencilSize * kMaxStencilSize];
namespace Kernels {

void convolve2D(const uint8_t *aSrc, uint8_t *aDst, size_t aRows, size_t aCols, size_t aStencilSize)
{
}
}; // namespace Kernels