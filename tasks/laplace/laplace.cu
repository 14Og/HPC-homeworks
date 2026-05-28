#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../check.hh"
#include "grid.hh"
#include "jacobi_solver.hh"

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

namespace Kernels {

__global__ void jacobi(const float* aCurr, float* aNext, int aNx, int aNy) {
    // TODO:
    // 1. Compute global 1-D index:  int i = blockIdx.x * blockDim.x + threadIdx.x
    // 2. Map to 2-D:  iy = i / aNx,  ix = i % aNx
    // 3. Skip if out of range (i >= aNx*aNy) or on any boundary (iy==0, iy==aNy-1,
    //    ix==0, ix==aNx-1)
    // 4. aNext[i] = 0.25f * (aCurr[(iy-1)*aNx+ix] + aCurr[(iy+1)*aNx+ix]
    //                      + aCurr[iy*aNx+(ix-1)] + aCurr[iy*aNx+(ix+1)])
}

__global__ void blockMaxDiff(const float* aCurr, const float* aNext,
                             float* blockMax, int aNx, int aNy) {
    extern __shared__ float sdata[];

    // TODO:
    // 1. Compute global index i and thread index tx = threadIdx.x
    // 2. Load into sdata[tx]:
    //      |curr[i] - next[i]| if i is an interior point, else 0.0f
    // 3. __syncthreads()
    // 4. Tree reduction — halve active threads each step:
    //      for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    //          if (tx < s) sdata[tx] = fmaxf(sdata[tx], sdata[tx + s]);
    //          __syncthreads();
    //      }
    // 5. if (tx == 0) blockMax[blockIdx.x] = sdata[0];
}

}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

void save_csv(const std::string& path, const Grid& g) {
    std::ofstream f(path);
    for (int iy = 0; iy < g.nY; ++iy) {
        for (int ix = 0; ix < g.nX; ++ix) {
            f << g.h_data[iy * g.nX + ix];
            if (ix + 1 < g.nX) f << ',';
        }
        f << '\n';
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    constexpr int   NX       = 256;
    constexpr int   NY       = 256;
    constexpr float TOL      = 1e-5f;
    constexpr int   MAX_ITER = 100'000;

    Grid g(NX, NY);

    // TODO: set boundary conditions in g.h_data[iy * NX + ix]
    // Boundaries are fixed for the entire solve — set them here, never touched again.
    //
    // Example layout (replace with the actual BCs from the problem image):
    //   top    (iy == NY-1):  g.h_data[(NY-1)*NX + ix] = ...
    //   bottom (iy == 0):     g.h_data[ix]              = ...
    //   left   (ix == 0):     g.h_data[iy * NX]         = ...
    //   right  (ix == NX-1):  g.h_data[iy * NX + NX-1] = ...

    g.upload();

    JacobiSolver solver(TOL, MAX_ITER);
    const int iters = solver.solve(g);
    std::printf("Converged in %d iterations\n", iters);

    g.download();
    save_csv("solution.csv", g);

    return 0;
}
