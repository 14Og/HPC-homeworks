#include <filesystem>
#include <iostream>

#include "histogram.cuh"

static inline std::filesystem::path blurDir(BLUR_DIR);
static inline std::filesystem::path histDir(HIST_DIR);
static inline std::filesystem::path grayDir(GRAY_DIR);

int main()
{
	std::filesystem::create_directories(histDir);
	std::filesystem::create_directories(grayDir);

	for (const auto &entry : std::filesystem::directory_iterator(blurDir)) {
		if (entry.path().extension() != ".jpg")
			continue;

		Histogram hist(entry.path());
		hist.make();

		auto stem = entry.path().stem().string();
		hist.save(histDir / (stem + "_hist.csv"));
		hist.saveGrayImage(grayDir / (stem + "_gs.jpg"));

		std::cout << "saved hist and grayscale for: " << entry.path() << std::endl;
	}
}