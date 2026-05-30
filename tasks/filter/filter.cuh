#include <cstdint>
#include <filesystem>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "check.cuh"
#include "cuda_image_wrapper.cuh"
#include "filter_kernels.cuh"

enum class FilterType : uint8_t { GAUSSIAN, BOX };

class Filter {
public:
	Filter(const std::filesystem::path &aPath) : src(cv::imread(aPath.string())), dst(src.img.rows, src.img.cols, CV_8UC3)
	{
	}

	~Filter()                         = default;
	Filter(const Filter &)            = delete;
	Filter &operator=(const Filter &) = delete;
	Filter(Filter &&)                 = delete;
	Filter &operator=(Filter &&)      = delete;

	void process(FilterType aType, uint8_t aStencilSize, size_t aBlockSize = 16)
	{
		if (aStencilSize < 3)
			throw std::runtime_error("Filter::process(): kernel size is too small");
		if (aStencilSize > kMaxStencilSize)
			throw std::runtime_error("Filter::process(): kernel size exceeded");
		if (!(aStencilSize % 2))
			throw std::runtime_error("Filter::process(): kernel size has to be odd");

		switch (aType) {
			case FilterType::BOX:
				fillBoxKernel(aStencilSize);
				break;
			case FilterType::GAUSSIAN:
				fillGaussianKernel(aStencilSize);
				break;
			default:
				throw std::runtime_error("Filter::process() wrong filter type");
		}

		src.upload();
		dim3 dimGrid((src.img.cols + aBlockSize - 1) / aBlockSize,
			(src.img.rows + aBlockSize - 1) / aBlockSize);
		dim3 dimBlock(aBlockSize, aBlockSize);

		Kernels::convolve2D<<<dimGrid, dimBlock>>>(
			src.dImg, dst.dImg, src.img.rows, src.img.cols, src.img.channels(), aStencilSize);
		CUDA_CHECK(cudaDeviceSynchronize());
		dst.download();
	}

	void saveProcessedImage(const std::filesystem::path &aPath)
	{
		if (dst.img.empty())
			throw std::runtime_error(
				"Filter::saveProcessedImage(): image has not yet been processed");

		cv::imwrite(aPath.string(), dst.img);
		std::cout << "saved processed image to: " << aPath.string() << std::endl;
	}

	cv::Mat getProcessedImage()
	{
		if (dst.img.empty())
			throw std::runtime_error(
				"Filter::getProcessedImage(): image has not yet been processed");

		return dst.img;
	}

private:
	void fillBoxKernel(size_t aStencilSize)
	{
		auto numElems = aStencilSize * aStencilSize;
		std::vector<float> kernelFlat(numElems);
		for (auto &elem : kernelFlat) elem = 1.0f / numElems;

		CUDA_CHECK(cudaMemcpyToSymbol(kernel, kernelFlat.data(), numElems * sizeof(float)));
	}

	void fillGaussianKernel(size_t aStencilSize)
	{
		auto numElems = aStencilSize * aStencilSize;
		std::vector<float> kernelFlat(numElems);
		float sigma = aStencilSize / 6.0f; // kernel covers +-3sigma
		int half    = aStencilSize / 2;
		float sum   = 0.0f;

		for (size_t i = 0; i < numElems; ++i) {
			auto y        = static_cast<int>(i / aStencilSize) - half;
			auto x        = static_cast<int>(i % aStencilSize) - half;
			float val     = std::exp(-(x * x + y * y) / (2.f * sigma * sigma));
			kernelFlat[i] = val;
			sum += val;
		}

		for (size_t i = 0; i < numElems; ++i) kernelFlat[i] /= sum;

		CUDA_CHECK(cudaMemcpyToSymbol(kernel, kernelFlat.data(), numElems * sizeof(float)));
	}

private:
	CudaImage src;
	CudaImage dst;

};