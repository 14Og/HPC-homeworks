#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <filesystem>

#include <cuda_runtime.h>

#include "check.cuh"
#include "grid.cuh"
#include "jacobi_solver.cuh"

static inline std::filesystem::path outDir{OUT_DIR};

void saveCSV(const std::string &path, const Grid &g)
{
	std::ofstream f(path);
	for (int iy = 0; iy < g.size(); ++iy) {
		for (int ix = 0; ix < g.size(); ++ix) {
			f << g[iy * g.size() + ix];
			if (ix + 1 < g.size())
				f << ',';
		}
		f << '\n';
	}
}

int main()
{
	static constexpr int kN{1024};
	static constexpr float kTol{1e-5f};
	static constexpr size_t kMaxIter{100'000};

	Grid g(kN, 0, 1, 0, 0);
	g.upload();

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start));
	JacobiSolver solver(kTol, kMaxIter, 1024);
	const int iters = solver.solve(g);
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	std::printf("Converged in %d iterations in %.1f ms\n", iters, ms);

	g.download();
	saveCSV(outDir / "laplace_result.csv", g);

	return 0;
}
