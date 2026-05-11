#include <mpi.h>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>

static constexpr int kN{100};
static constexpr int kTagBall{0xBA};
static constexpr int kTagDone{0xD0};

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int rank{0};
	int size{0};
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size < 2) {
		std::cerr << "comm size is too small: " << size << std::endl;
		MPI_Finalize();
		return 1;
	}

	// Seed random generator differently for each process
	std::mt19937 gen(rank + std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_int_distribution<> dist(0, size - 1);

	// Process 0 initiates the game
	if (rank == 0) {
		std::vector<int> message;
		message.push_back(0);

		int nextRank = dist(gen);
		if (size > 1) {
			while (nextRank == rank) nextRank = dist(gen);
		}

		MPI_Ssend(message.data(), message.size(), MPI_INT, nextRank, kTagBall, MPI_COMM_WORLD);
	}

	bool done = false;
	std::vector<int> buffer(kN + 1);

	while (!done) {
		MPI_Status status;
		MPI_Recv(buffer.data(), buffer.size(), MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
			&status);

		if (status.MPI_TAG == kTagDone) {
			done = true;
		} else if (status.MPI_TAG == kTagBall) {
			int count{0};
			MPI_Get_count(&status, MPI_INT, &count);

			// Append my rank
			buffer[count] = rank;

			if (count == kN) {
				// We reached the final pass
				std::cout << "Game over! Rank " << rank << " received the " << kN
						  << "th pass.\nFinal sequence: ";
				for (int i = 0; i <= count; ++i) std::cout << buffer[i] << " ";

				std::cout << std::endl;

				// Notify all other processes to terminate
				int dummy{1};
				for (int i = 0; i < size; ++i) {
					if (i != rank) {
						MPI_Ssend(&dummy, 1, MPI_INT, i, kTagDone, MPI_COMM_WORLD);
					}
				}
				done = true;
			} else {
				// Continue passing
				int nextRank = dist(gen);
				if (size > 1) {
					while (nextRank == rank) nextRank = dist(gen);
				}
				MPI_Ssend(buffer.data(), count + 1, MPI_INT, nextRank, kTagBall, MPI_COMM_WORLD);
			}
		}
	}

	MPI_Finalize();
	return 0;
}
