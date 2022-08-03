#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "utils.h"
#include <vector>
#include <algorithm>
#include <curand.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>

//泊松采样根据Wei et al.的 Parallel Poisson Disk Sampling 实现，并未进行充分优化

int doPoissonSample(std::vector<int>& seedArr, dim3 center, int centerRadius, int width, int height, int slice,int newSize, uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress);


int doPoissonSample_cpu(std::vector<int>& seedArr, dim3 center, int centerRadius, int width, int height, int slice, int newSize, uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress);


int doPoissonSample2(std::vector<int>& seedArr, dim3 center, int centerRadius, int width, int height, int slice, int newSize, uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress);

int readPoissonSample(const std::string poisson_sample_dir, std::vector<int>& seedArr, dim3 center, int centerRadius, int width, int height, int slice,
	const uchar * imagePtr);

int filterPoissonSample(uchar* imagePtr, std::vector<float> &posX, std::vector<float>& posY, std::vector<float>& posZ,
	std::vector<int>& seedArr, dim3 center, int centerRadius, int width, int height, int slice);

void filterPoissonSample_gpu(std::vector<int>& seedArr, dim3 center, int centerRadius, int sampleNum,
	int width, int height, int slice, uchar* d_imagePtr, float* d_X_new_out, float* d_Y_new_out, float* d_Z_new_out);
