#include "Eigen/Dense"
#include "omp.h"

#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using Eigen::Index;
using Eigen::VectorXd;

struct LineParams {
	double a;
	double b;
};

std::pair<VectorXd, VectorXd> makeNoisyLinearData(Index aN, LineParams aTrue, double aNoise, size_t aSeed = 42)
{
	std::srand(aSeed);
	VectorXd x     = VectorXd::Random(aN); // uniform [-1, 1]
	VectorXd noise = VectorXd::Random(aN) * aNoise;
	VectorXd y     = aTrue.a * x.array() + aTrue.b + noise.array();
	return {std::move(x), std::move(y)};
}

std::optional<LineParams> gradientDescent(
	const VectorXd &aX, const VectorXd &aY, double aLr = 0.1, double aTol = 1e-6, int aMaxIter = 100000)
{
	const Index n = aX.size();
	double a      = 0.0;
	double b      = 0.0;

	for (int it = 0; it < aMaxIter; ++it) {
		double gradA = 0.0;
		double gradB = 0.0;

#pragma omp parallel for reduction(+ : gradA, gradB) schedule(static)
		for (Index i = 0; i < n; ++i) {
			double r = a * aX(i) + b - aY(i);
			gradA += aX(i) * r;
			gradB += r;
		}

		gradA *= 2.0 / static_cast<double>(n);
		gradB *= 2.0 / static_cast<double>(n);

		a -= aLr * gradA;
		b -= aLr * gradB;

		if (std::sqrt(gradA * gradA + gradB * gradB) < aTol) {
			std::cout << "converged in " << it + 1 << " iterations\n";
			return LineParams{a, b};
		}
	}
	return {};
}

void benchmark(const VectorXd &aX, const VectorXd &aY)
{
	[[maybe_unused]] auto t0 = omp_get_wtime();
	auto line                = gradientDescent(aX, aY);
	[[maybe_unused]] auto t1 = omp_get_wtime();
	if (!line) {
		std::cerr << "gradient descent did not converge" << std::endl;
		return;
	}
	std::cout << "threads=" << omp_get_max_threads() << "  time=" << (t1 - t0) << " s" << "  a=" << line->a
			  << "  b=" << line->b << "\n";
}

int main(int argc, char **argv)
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

	auto [x, y] = makeNoisyLinearData(1'000'000, {2.5, -1.0}, 0.1);

	for (int t : threadCounts) {
		omp_set_num_threads(t);
		benchmark(x, y);
	}
}
