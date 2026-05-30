#include <filesystem>

#include "filter.cuh"

static inline std::filesystem::path blurDir{BLUR_DIR};
static inline std::filesystem::path mediaDir{MEDIA_DIR};

__constant__ float kernel[kMaxStencilSize * kMaxStencilSize];

int main(int argc, char **argv)
{
	Filter filter(mediaDir / "doom.jpg");

	for (int i = 11; i <= kMaxStencilSize; i += 20) {
		std::string boxRepl   = "doom_boxfilter";
		std::string gaussRepl = "doom_gaussian";

		boxRepl += "_" + std::to_string(i);
		gaussRepl += "_" + std::to_string(i);

		filter.process(FilterType::BOX, i);
		filter.saveProcessedImage((blurDir / boxRepl).replace_extension("jpg"));

		filter.process(FilterType::GAUSSIAN, i);
		filter.saveProcessedImage((blurDir / gaussRepl).replace_extension("jpg"));
	}
}