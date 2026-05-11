#include <mpi.h>

#include <iostream>
#include <vector>

static constexpr int kMaxPower   = 20; // Up to 4 megabytes
static constexpr int kIterations = 10000;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int rank{0};
	int size{0};
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size != 2) {
		if (rank == 0) {
			std::cerr << "This program requires exactly 2 processes." << std::endl;
		}
		MPI_Finalize();
		return 1;
	}

	if (rank == 0) {
		std::cout << "Message_Size_Bytes,Time_Per_Pass_Sec,Bandwidth_MB_s\n";
	}

	for (int power = -1; power <= kMaxPower; ++power) {
		// Element count: 0 for power=-1, else 2^power
		int count            = (power == -1) ? 0 : (1 << power);
		int messageSizeBytes = count * sizeof(int);

		std::vector<int> buffer(count, 1);

		// Synchronize before starting the timer
		MPI_Barrier(MPI_COMM_WORLD);

		double startTime = MPI_Wtime();

		for (int i = 0; i < kIterations; ++i) {
			if (rank == 0) {
				MPI_Ssend(buffer.data(), count, MPI_INT, 1, 0, MPI_COMM_WORLD);
				MPI_Recv(buffer.data(), count, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			} else if (rank == 1) {
				MPI_Recv(buffer.data(), count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Ssend(buffer.data(), count, MPI_INT, 0, 0, MPI_COMM_WORLD);
			}
		}

		double endTime = MPI_Wtime();

		if (rank == 0) {
			double timePerPass = (endTime - startTime) / (2.0 * kIterations);
			double bandwidth = 0.0;

			if (timePerPass > 0.0) {
				bandwidth = (messageSizeBytes / timePerPass) / (1024.0 * 1024.0);
			}

			std::cout << messageSizeBytes << "," << timePerPass << "," << bandwidth << "\n";
		}
	}

	MPI_Finalize();
	return 0;
}
