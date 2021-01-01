#pragma once

#include <iostream>
#include <stdint.h>

#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR                                                       \
    do {                                                                       \
        const cudaError_t err = cudaGetLastError();                            \
        if (err != cudaSuccess) {                                              \
            const char *const err_str = cudaGetErrorString(err);               \
            std::cerr << "Cuda error in " << __FILE__ << ":" << __LINE__ - 1   \
                      << ": " << err_str << " (" << err << ")" << std::endl;   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while(0)


inline unsigned int div_up(unsigned int numerator, unsigned int denominator)
{
	unsigned int result = numerator / denominator;
	if (numerator % denominator) ++result;
	return result;
}

struct filterkernel_gpu {
	int ks;
	float *data;

#ifndef __CUDACC__
	// If you want, you can implement this stuff
	filterkernel_gpu(const filterkernel_gpu&) = delete;
	filterkernel_gpu &operator=(const filterkernel_gpu&) = delete;
#endif

	filterkernel_gpu(int ks);
	~filterkernel_gpu();
};

struct filterkernel_cpu {
	int ks;
	float *data;

#ifndef __CUDACC__
	// If you want, you can implement this stuff
	filterkernel_cpu(const filterkernel_cpu&) = delete;
	filterkernel_cpu &operator=(const filterkernel_cpu&) = delete;
#endif

	filterkernel_cpu(int size);
	~filterkernel_cpu()
	{
		delete[] data;
	}

	void upload(filterkernel_gpu &dst) const;
	void upload_cmem() const;
};
