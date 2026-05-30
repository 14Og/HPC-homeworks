#ifndef TASKS_HISTOGRAM_HISTOGRAM_CUH_
#define TASKS_HISTOGRAM_HISTOGRAM_CUH_

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "histogram_kernels.cuh"
#include "cuda_image_wrapper.cuh"

class Histogram {
	static constexpr size_t kHistElems{256};
	static constexpr size_t kHistSize{kHistElems * sizeof(int)};

public:
	Histogram(const std::filesystem::path &aPath) :
		src(cv::imread(aPath.string())),
		grayscale(src.img.rows, src.img.cols, CV_8UC1),
		hist(kHistElems)
	{
		CUDA_CHECK(cudaMalloc(&dHist, kHistSize));
	}

	~Histogram()
	{
		if (dHist)
			cudaFree(dHist);
	}

	Histogram(const Histogram &)            = delete;
	Histogram &operator=(const Histogram &) = delete;
	Histogram(Histogram &&)                 = delete;
	Histogram &operator=(Histogram &&)      = delete;

	void make(size_t aBlockSize = 16)
	{
		if (src.img.empty())
			throw std::runtime_error("Histogram::make(): empty source image");
		toGrayscale(aBlockSize);
		makeHist(aBlockSize);
	}

	void save(const std::filesystem::path &aPath)
	{
		std::ofstream f(aPath);
		if (!f)
			throw std::runtime_error("Histogram::save(): cannot open " + aPath.string());
		f << "bin,count\n";
		for (size_t i = 0; i < hist.size(); ++i)
			f << i << ',' << hist[i] << '\n';
	}

    void saveGrayImage(const std::filesystem::path &aPath)
    {
        if (grayscale.img.empty())
            throw std::runtime_error("Histogram::saveGrayscale(): empty grayscale image");

        cv::imwrite(aPath.string(), grayscale.img);
    }

private:
	void toGrayscale(size_t aBlockSize)
	{
		src.upload();
		dim3 dimGrid((src.img.cols + aBlockSize - 1) / aBlockSize,
			(src.img.rows + aBlockSize - 1) / aBlockSize);
		dim3 dimBlock(aBlockSize, aBlockSize);
		Kernels::BGR2Grayscale<<<dimGrid, dimBlock>>>(
			src.dImg, grayscale.dImg, src.img.rows, src.img.cols);
		grayscale.download();
	}

	void makeHist(size_t aBlockSize)
	{
		CUDA_CHECK(cudaMemset(dHist, 0, kHistSize));
		dim3 dimGrid((grayscale.img.cols + aBlockSize - 1) / aBlockSize,
			(grayscale.img.rows + aBlockSize - 1) / aBlockSize);
		dim3 dimBlock(aBlockSize, aBlockSize);
		Kernels::histogram<<<dimGrid, dimBlock>>>(
			grayscale.dImg, grayscale.img.rows, grayscale.img.cols, dHist);
		CUDA_CHECK(cudaMemcpy(hist.data(), dHist, kHistSize, cudaMemcpyDeviceToHost));
	}

private:
	CudaImage src;
	CudaImage grayscale;

	std::vector<int> hist;
	int *dHist{nullptr};
};

#endif /* TASKS_HISTOGRAM_HISTOGRAM_CUH_ */
