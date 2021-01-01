#include <stdexcept>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>

#include "common.h"

filterkernel_gpu::filterkernel_gpu(int size) : ks(size)
{
	size_t size_t = size;
	cudaError_t err = cudaMalloc(&(this->data),size_t*sizeof(float));
}
filterkernel_gpu::~filterkernel_gpu()
{
	//cudaError_t err = cudaFree(this->data);
}

void filterkernel_cpu::upload(filterkernel_gpu &dst) const
{
	cudaMemcpy(dst.data, this->data, this->ks*sizeof(float), cudaMemcpyHostToDevice);
}

// Initializes the given 1D kernel with a univariate Gaussian
filterkernel_cpu::filterkernel_cpu(int size) : ks(size), data(new float[size])
{
	if ((ks & 1) != 1 || ks < 1)
		throw std::runtime_error("Invalid kernel size");

	// Set the mean to lie in the middle of the kernel
	const float mu = (ks - 1) / 2.0f;

	// sigma is chosen so that the smallest kernel (k=3) will span +-0.8 sigma
	// of the normal distribution.
	// For larger kernels (k -> inf) sigma is chosen so that they span +-3
	// sigma.
	// Credit goes to OpenCV.
	const float sigma = (ks - 1) / (2.0f * 3.0f) + 0.8f;

	// PDF of the normal distribution
	const auto gaussian = [mu, sigma](const float x) -> float {
		const float v = x - mu;
		return std::exp(-v * v / (2.0 * sigma * sigma)) /
				(sigma * std::sqrt(2.0 * M_PI));
	};

	// Generate the kernel by point sampling the PDF at the pixel centers.
	// There are much better methods to do this but we don't need them here.
	float sum = 0.0f;
	for (int i = 0; i < ks; i++) {
		float temp = gaussian(i);
		data[i] = temp;
		sum += temp;
	}

	// Normalize kernel values
	const float rcpSum = 1.0f / sum;
	for (int i = 0; i < ks; i++) {
		data[i] *= rcpSum;
	}
}
