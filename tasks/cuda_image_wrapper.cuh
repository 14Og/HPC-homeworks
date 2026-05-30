#ifndef TASKS_CUDA_IMAGE_WRAPPER_CUH_
#define TASKS_CUDA_IMAGE_WRAPPER_CUH_

#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "check.cuh"

struct CudaImage {
	cv::Mat img;
	uint8_t *dImg{nullptr};

	template<typename... Args>
	CudaImage(Args &&...aArgs) : img(std::forward<Args>(aArgs)...)
	{
		CUDA_CHECK(cudaMalloc(&dImg, numBytes()));
	}

	~CudaImage()
	{
		if (dImg)
			cudaFree(dImg);
	}

	CudaImage(const CudaImage &)            = delete;
	CudaImage &operator=(const CudaImage &) = delete;
	CudaImage(CudaImage &&)                 = delete;
	CudaImage &operator=(CudaImage &&)      = delete;

	void upload()
	{
		CUDA_CHECK(cudaMemcpy(dImg, img.data, numBytes(), cudaMemcpyHostToDevice));
	}

	void download()
	{
		CUDA_CHECK(cudaMemcpy(img.data, dImg, numBytes(), cudaMemcpyDeviceToHost));
	}

	size_t numBytes()
	{
		return img.total() * img.elemSize();
	}
};

#endif /* TASKS_CUDA_IMAGE_WRAPPER_CUH_ */
