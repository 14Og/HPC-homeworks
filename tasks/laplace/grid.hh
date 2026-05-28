#ifndef TASKS_LAPLACE_GRID_HH_
#define TASKS_LAPLACE_GRID_HH_

#include <cstdlib>
#include <vector>
#include <array>

#include <cuda_runtime.h>

#include "check.hh"

// 2D grid owning two ping-pong device buffers and a host-side copy.
//
// Convention:
//   n  — number of grid points per side (square: same for x and y)
//   element (iy, ix) is at h_data[iy * n + ix]
//
// Typical workflow:
//   1. Grid g(n)         — allocates device memory
//   2. fill g.h_data     — set boundary conditions in main()
//   3. g.upload()        — copies h_data to both device buffers
//   4. solver.solve(g)
//   5. g.download()      — copies result back to h_data

class Grid {

public:
	Grid(int aN, float aTop, float aBottom, float aLeft, float aRight) : n(aN), hData(n * n, 0.0f)
	{
		for (auto &pointer : dBuffers) CUDA_CHECK(cudaMalloc(&pointer, bufSize()));
		fill(aTop, aBottom, aLeft, aRight);
	}

	~Grid()
	{
		for (auto pointer : dBuffers)
			if (pointer)
				CUDA_CHECK(cudaFree(pointer));
	}

	Grid(const Grid &)            = delete;
	Grid &operator=(const Grid &) = delete;
	Grid(Grid &&)                 = delete;
	Grid &operator=(Grid &&)      = delete;

	size_t size() const
	{
		return n;
	}

	float &operator[](size_t aId)
	{
		return hData[aId];
	}

	const float &operator[](size_t aId) const
	{
		return hData[aId];
	}

	void upload()
	{
		for (auto pointer : dBuffers) CUDA_CHECK(cudaMemcpy(pointer, hData.data(), bufSize(), cudaMemcpyHostToDevice));
	}

	void download()
	{
		for (auto pointer : dBuffers) CUDA_CHECK(cudaMemcpy(hData.data(), pointer, bufSize(), cudaMemcpyDeviceToHost));
	}

	float *devCurr()
	{
		return dBuffers[currIdx];
	}
	float *devNext()
	{
		return dBuffers[1 - currIdx];
	}
	void swapBuffers()
	{
		currIdx ^= 1;
	}

private:
	void fill(float aTop, float aBottom, float aLeft, float aRight)
	{
		std::fill(hData.begin(), hData.begin() + n, aBottom); // bottom fill
		std::fill(hData.end() - n, hData.end(), aTop); // top fill

		for (int i = 0; i < n * n; i += n) hData[i] = aLeft; // left fill
		for (int i = n - 1; i < n * n; i += n) hData[i] = aRight; // right fill

		// averaging ambiguous corners
		hData[0]           = (aBottom + aLeft) / 2;
		hData[n - 1]       = (aBottom + aRight) / 2;
		hData[n * (n - 1)] = (aTop + aLeft) / 2;
		hData[n * n - 1]   = (aTop + aRight) / 2;
	}

	size_t bufSize()
	{
		return n * n * sizeof(float);
	}

private:
	size_t n{0};
	std::vector<float> hData;
	std::array<float *, 2> dBuffers{nullptr, nullptr};
	size_t currIdx{0};
};

#endif /* TASKS_LAPLACE_GRID_HH_ */
