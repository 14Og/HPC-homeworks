#ifndef TASKS_LAPLACE_GRID_HH_
#define TASKS_LAPLACE_GRID_HH_

#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "check.hh"

// 2D grid owning two ping-pong device buffers and a host-side copy.
//
// Convention:
//   nX  — number of grid points in x (columns)
//   nY  — number of grid points in y (rows)
//   element (iy, ix) is at h_data[iy * nX + ix]
//
// Typical workflow:
//   1. Grid g(nX, nY)    — allocates device memory
//   2. fill g.h_data     — set boundary conditions in main()
//   3. g.upload()        — copies h_data to both device buffers
//   4. solver.solve(g)
//   5. g.download()      — copies result back to h_data

class Grid {
public:
	Grid(int nX, int nY) : nX(nX), nY(nY), h_data(nX * nY, 0.0f), curr(0)
	{
		// TODO: cudaMalloc both dBuffers[0] and dBuffers[1]  (size = nX * nY * sizeof(float))
	}

	~Grid()
	{
		// TODO: cudaFree both device buffers
	}

	// Copies h_data into both device buffers (so both start identical)
	void upload()
	{
		// TODO: cudaMemcpy h_data.data() -> dBuffers[0] and dBuffers[1]  (HostToDevice)
	}

	// Copies the current device buffer back to h_data
	void download()
	{
		// TODO: cudaMemcpy dBuffers[curr] -> h_data.data()  (DeviceToHost)
	}

	float *devCurr()
	{
		return dBuffers[curr];
	}
	float *devNext()
	{
		return dBuffers[1 - curr];
	}
	void swapBuffers()
	{
		curr ^= 1;
	}

	Grid(const Grid &)            = delete;
	Grid &operator=(const Grid &) = delete;

public:
	const int nX, nY;
	std::vector<float> h_data; // host buffer; fill BCs here before upload()

private:
	float *dBuffers[2];
	int curr;
};

#endif /* TASKS_LAPLACE_GRID_HH_ */
