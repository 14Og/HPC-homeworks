#include "omp.h"

#include <format>
#include <iostream>
#include <string>
#include <vector>

double computePi(size_t aN)
{
	double sum = 0;
	auto step  = 1 / static_cast<double>(aN);

#pragma omp parallel for reduction(+ : sum) schedule(static)
	for (int i = 0; i < static_cast<int>(aN); ++i) {
		auto x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}

	return sum * step;
}

void benchmark(size_t aN, size_t aNthreads)
{
	if (aNthreads < 1)
		throw std::runtime_error("wrong threads num");

	omp_set_num_threads(aNthreads);

	[[maybe_unused]] auto t0 = omp_get_wtime();
	auto pi                  = computePi(aN);
	[[maybe_unused]] auto t1 = omp_get_wtime();
	std::cout << std::format("threads={}  time={} s  pi={:.25f}\n", omp_get_max_threads(), t1 - t0, pi);
}

int main(int argc, char **argv)
{
	std::vector<int> threadCounts;
	threadCounts.reserve(argc - 1);
	try {
		for (int i = 1; i < argc; ++i) threadCounts.push_back(std::stoi(argv[i]));
	} catch (...) {
		std::cerr << "wrong cli arguments passed" << std::endl;
		std::cerr << "usage: " << argv[0] << " num_threads1 num_threads2 num_threads3 ..." << std::endl;
		throw;
	}
	if (threadCounts.empty())
		threadCounts.push_back(omp_get_max_threads());

	static constexpr size_t kN{100'000'000};

	for (auto count : threadCounts) {
		benchmark(kN, count);
	}
}
