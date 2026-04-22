#include "Eigen/Dense"
#include "omp.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <optional>
#include <vector>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

std::pair<MatrixXd, VectorXd> makeStrictlyDiagonallyDominantSystem(Index aN, size_t aSeed = 42)
{
	std::srand(aSeed);
	MatrixXd A = MatrixXd::Random(aN, aN);
	VectorXd b = VectorXd::Random(aN) * aN;

	for (Index i = 0; i < aN; ++i) {
		auto offSum = A.row(i).cwiseAbs().sum() - std::abs(A(i, i));
		A(i, i)     = offSum + 1;
	}

	return {std::move(A), std::move(b)};
}

std::optional<VectorXd> jacobi(const MatrixXd &aA, const VectorXd &aB, double aTol = 1e-8, int aMaxIter = 10000)
{
	const Index n = aA.rows();
	VectorXd xOld = VectorXd::Zero(n);
	VectorXd xNew(n);

	for (int it = 0; it < aMaxIter; ++it) {
		double diffSq = 0.0;

#pragma omp parallel for reduction(+ : diffSq) schedule(static)
		for (Index i = 0; i < n; ++i) {
			double sum = aA.row(i).dot(xOld) - aA(i, i) * xOld(i);
			xNew(i)    = (aB(i) - sum) / aA(i, i);
			double d   = xNew(i) - xOld(i);
			diffSq += d * d;
		}

		xOld.swap(xNew);

		if (std::sqrt(diffSq) < aTol)
			return {std::move(xOld)};
	}
	return {};
}

void benchmark(const MatrixXd &aA, const VectorXd &aB, size_t aNthreads)
{
	if (aNthreads < 1)
		throw std::runtime_error("wrong num threads");

	omp_set_num_threads(aNthreads);

	[[maybe_unused]] auto t0 = omp_get_wtime();
	auto x                   = jacobi(aA, aB);
	[[maybe_unused]] auto t1 = omp_get_wtime();
	if (!x) {
		std::cerr << "jacobi did not converge" << std::endl;
		return;
	}
	std::cout << "threads:  " << omp_get_max_threads() << "\n";
	std::cout << "time:     " << t1 - t0 << " s\n";
	std::cout << "residual: " << (aA * x.value() - aB).norm() << "\n";
	std::cout << std::endl;
}

int main(int argc, char **argv, [[maybe_unused]] char **envp)
{
	Eigen::setNbThreads(1); // disable internal eigen omp parallelization

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

	auto [A, b] = makeStrictlyDiagonallyDominantSystem(2000);
	for (auto count : threadCounts) benchmark(A, b, count);
}