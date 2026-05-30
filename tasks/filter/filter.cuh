#include "filter_kernels.cuh"

#include <cstdint>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

enum class FilterType : uint8_t { GAUSSIAN, BOX };

class Filter {
public:
	Filter(const std::filesystem::path &aPath)
	{
		sourceImage = cv::imread(aPath.string());
	}

	Filter(const Filter &)            = delete;
	Filter &operator=(const Filter &) = delete;
	Filter(Filter &&)                 = delete;
	Filter &operator=(Filter &&)      = delete;


    void process(FilterType aType, uint8_t aKernelSize)
    {
        
    }

    cv::Mat getProcessedImage()
    {
        return processedImage;
    }

private:
	cv::Mat sourceImage;
    cv::Mat processedImage;
};