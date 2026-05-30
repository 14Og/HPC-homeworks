#include <cstdint>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "filter_kernels.cuh"
#include "check.hh"


enum class FilterType : uint8_t { GAUSSIAN, BOX };

class Filter {
public:
	Filter(const std::filesystem::path &aPath)
	{
		imgSrc = cv::imread(aPath.string());
		numBytes = imgSrc.total() * imgSrc.elemSize();
		upload();
	}

	Filter(const Filter &)            = delete;
	Filter &operator=(const Filter &) = delete;
	Filter(Filter &&)                 = delete;
	Filter &operator=(Filter &&)      = delete;


    void process(FilterType aType, uint8_t aStencilSize)
    {
		if (aStencilSize > kMaxStencilSize)
			throw std::runtime_error("Filter::process(): kernel size exceeded");
    }

    cv::Mat getProcessedImage()
    {
		if (imgProcessed.empty())
			throw std::runtime_error("Filter::getProcessedImage(): image has not yet been processed");

        return imgProcessed;
    }

private:

	// allocate and copy image to device memory
	void upload()
	{
		CUDA_CHECK(cudaMalloc(&dImgSrc, numBytes));
		CUDA_CHECK(cudaMalloc(&dImgProcessed, numBytes));
		CUDA_CHECK(cudaMemcpy(dImgSrc, imgSrc.data, numBytes, cudaMemcpyHostToDevice));
		
	}

	// copy processed image from device to host memory
	void download()
	{
		imgProcessed = cv::Mat(imgSrc.rows, imgSrc.cols, imgSrc.type());
		CUDA_CHECK(cudaMemcpy(imgProcessed.data, dImgProcessed, numBytes, cudaMemcpyDeviceToHost));
	}

private:
	cv::Mat imgSrc;
    cv::Mat imgProcessed;

	uint8_t *dImgSrc{nullptr};
	uint8_t *dImgProcessed{nullptr};

	size_t numBytes{0};
};