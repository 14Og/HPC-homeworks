#include <mpi.h>

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int rank{0}, size{0};
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Default parameters
	int rule       = 30;
	bool periodic  = false;
	int totalCells = 100;
	int steps      = 50;
	bool printGrid = false;

	if (argc >= 3) {
		rule              = std::atoi(argv[1]);
		std::string bcond = argv[2];
		if (bcond == "--periodic") {
			periodic = true;
		} else if (bcond == "--static") {
			periodic = false;
		} else {
			if (rank == 0)
				std::cerr << "Unknown boundary condition. Use --static or --periodic.\n";
			MPI_Finalize();
			return 1;
		}
	}
	if (argc >= 4)
		totalCells = std::atoi(argv[3]);
	if (argc >= 5)
		steps = std::atoi(argv[4]);
	if (argc >= 6)
		printGrid = (std::atoi(argv[5]) != 0);

	if (rank == 0 && printGrid) {
		std::cout << "Automata 1D - Rule " << rule << "\n";
		std::cout << "Boundary: " << (periodic ? "Periodic" : "Static") << "\n";
		std::cout << "Cells: " << totalCells << ", Steps: " << steps << "\n\n";
	}

	// Domain decomposition
	int baseChunk   = totalCells / size;
	int remainder   = totalCells % size;

	int localSize   = baseChunk + (rank < remainder ? 1 : 0);
	int globalStart = rank * baseChunk + std::min(rank, remainder);

	// Buffer with 2 ghost cells
	std::vector<int> current(localSize + 2, 0);
	std::vector<int> next(localSize + 2, 0);

	int globalCenter = totalCells / 2;
	if (globalCenter >= globalStart && globalCenter < globalStart + localSize) {
		int localIndex      = globalCenter - globalStart + 1;
		current[localIndex] = 1;
	}

	// Neighbor ranks
	int leftNeighbor  = rank - 1;
	int rightNeighbor = rank + 1;

	if (periodic) {
		if (leftNeighbor < 0)
			leftNeighbor = size - 1;
		if (rightNeighbor >= size)
			rightNeighbor = 0;
	} else {
		if (leftNeighbor < 0)
			leftNeighbor = MPI_PROC_NULL;
		if (rightNeighbor >= size)
			rightNeighbor = MPI_PROC_NULL;
	}

	std::vector<int> recvCounts, displs;
	std::vector<int> fullGrid;
	std::ofstream outFile;

	if (printGrid) {
		if (rank == 0) {
			recvCounts.resize(size);
			displs.resize(size);
			fullGrid.resize(totalCells);
			outFile.open("assets/automata_history.csv");
			if (!outFile.is_open()) {
				std::cerr << "Error: Could not open automata_history.csv for writing.\n";
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
		MPI_Gather(&localSize, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (rank == 0) {
			displs[0] = 0;
			for (int i = 1; i < size; ++i) {
				displs[i] = displs[i - 1] + recvCounts[i - 1];
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double startTime = MPI_Wtime();

	for (int step = 0; step < steps; ++step) {
		if (printGrid) {
			// Gather inner array part (without ghost cells)
			MPI_Gatherv(&current[1], localSize, MPI_INT, fullGrid.data(), recvCounts.data(),
				displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

			if (rank == 0) {
				for (int i = 0; i < totalCells; ++i) {
					outFile << fullGrid[i] << (i == totalCells - 1 ? "" : ",");
				}
				outFile << "\n";
			}
		}

		// Enforce static boundaries on ghost cells specifically
		// If periodic, MPI_Sendrecv will overwrite these naturally.
		current[0]             = 0;
		current[localSize + 1] = 0;

		// Send rightmost true cell, receive left ghost cell
		MPI_Sendrecv(&current[localSize], 1, MPI_INT, rightNeighbor, 0, // sending
			&current[0], 1, MPI_INT, leftNeighbor, 0, // receiving
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// Send leftmost true cell, receive right ghost cell
		MPI_Sendrecv(&current[1], 1, MPI_INT, leftNeighbor, 1, // sending
			&current[localSize + 1], 1, MPI_INT, rightNeighbor, 1, // receiving
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// Compute next state
		for (int i = 1; i <= localSize; ++i) {
			// Extract 3-bit context: Left (bit 2), Center (bit 1), Right (bit 0)
			int context = (current[i - 1] << 2) | (current[i] << 1) | current[i + 1];
			next[i]     = (rule >> context) & 1;
		}

		// Swap buffers
		std::swap(current, next);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double endTime = MPI_Wtime();

	if (rank == 0) {
		std::cout << "\nTotal Simulation Time: " << (endTime - startTime) << " seconds\n";
		if (printGrid) {
			outFile.close();
		}
	}

	MPI_Finalize();
	return 0;
}