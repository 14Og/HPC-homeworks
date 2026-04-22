#include "omp.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <filesystem>
#include <atomic>

static constexpr auto kFormat{"P3"};
static inline std::filesystem::path media(MEDIA_DIR);

struct PPMPixel {
	uint8_t red{0};
	uint8_t green{0};
	uint8_t blue{0};
};

struct PPMImage {
	size_t x{0};
	size_t y{0};
	size_t all{0};
	std::vector<PPMPixel> data;
};

PPMImage readPPM(std::filesystem::path aPath)
{
	std::ifstream file(aPath);
	if (!file)
		throw std::runtime_error("no file at " + aPath.string());

	PPMImage img;
	std::string format;
	file >> format;
	if (format != kFormat)
		throw std::runtime_error("wrong format: " + format);

	file >> img.x >> img.y;
	[[maybe_unused]] int maxColor;
	file >> maxColor;
	img.all = img.x * img.y;
	img.data.reserve(img.all);
	for (size_t i = 0; i < img.all; i++) {
		int r, g, b;
		file >> r >> g >> b;
		img.data.emplace_back(static_cast<uint8_t>(r), static_cast<uint8_t>(g), static_cast<uint8_t>(b));
	}
	file.close();
	return img;
}

void animatePPM(std::filesystem::path aFramePath, const PPMImage &aImg, size_t aShifts)
{
	static constexpr size_t kRGB{255};
	std::atomic_bool errorFlag{false};
	if (!std::filesystem::exists(aFramePath))
		std::filesystem::create_directory(aFramePath);

#pragma omp parallel for shared(errorFlag)
	for (size_t shift = 0; shift < aShifts; ++shift) {
		std::ofstream file(aFramePath / std::filesystem::path("frame_" + std::to_string(shift)));
		if (!file || errorFlag) {
			errorFlag.store(true);
			continue;
		}

		file << kFormat << std::endl;
		file << aImg.x << " " << aImg.y << std::endl;
		file << kRGB << std::endl;

		for (size_t y = 0; y < aImg.y; ++y) {
			for (size_t x = 0; x < aImg.x; ++x) {
				auto srcNum     = (x + aImg.x - shift % aImg.x) % aImg.x;
				const auto &src = aImg.data[y * aImg.x + srcNum];
				file << static_cast<int>(src.red) << " " << static_cast<int>(src.green) << " "
					 << static_cast<int>(src.blue) << " ";
			}
			file << "\n";
		}
	}
	if (errorFlag)
		throw std::runtime_error("error during file creation!");
}

void benchmark(std::filesystem::path aFramePath, const PPMImage &aImg, size_t aShifts, size_t aNthreads)
{
	if (aNthreads < 1)
		throw std::runtime_error("wrong num threads");
	omp_set_num_threads(aNthreads);

	[[maybe_unused]] auto t0 = omp_get_wtime();
	animatePPM(std::move(aFramePath), aImg, aShifts);
	[[maybe_unused]] auto t1 = omp_get_wtime();
	std::cout << "threads:  " << omp_get_max_threads() << "\n";
	std::cout << "time:     " << t1 - t0 << " s\n";
	std::cout << std::endl;
}

int main(int argc, char **argv, [[maybe_unused]] char **envp)
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

	auto img = readPPM(media / "car.ppm");

	std::cout << "shifts: " << img.x << std::endl;
	for (auto count : threadCounts) benchmark(media / "frames", img, img.x, count);
}
