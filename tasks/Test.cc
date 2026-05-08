#include <mpi.h>

#include <iostream>

int main()
{
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	int worldSize{0};
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

	// Get the rank of the process
	int worldRank{0};
	MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

	// Get the name of the processor
	char procName[MPI_MAX_PROCESSOR_NAME];
	int nameLen{0};
	MPI_Get_processor_name(procName, &nameLen);

	// Print off a hello world message
	std::cout << "Hello world from processor " << procName << " rank " << worldRank << " out of "
			  << worldSize << " processors\n";

	// Finalize the MPI environment.
	MPI_Finalize();
}