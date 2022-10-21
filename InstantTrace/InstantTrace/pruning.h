#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "utils.h"
#include <map>
#include "TimerClock.hpp"
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>


inline float gIFunc(float value, float maxvalue = 255, float lambda = 10)
{
	return exp(lambda * (1 - value * 1.0f / maxvalue) * (1 - value * 1.0f / maxvalue));
	//gI(x) = exp(\lambda_i * (1 - I(x)/Imax) ** 2)
}

inline float gwdtFunc(float dist, float value1, float value2)
{
	return dist * (gIFunc(value1) + gIFunc(value2)) / 2;
	//e(x,y) = |x-y| * (gI(x) + gI(y)) / 2
}


void pruneLeaf_3d_gpu(std::vector<int>& leafArr, int &validLeafCount, std::vector<int>& disjointSet, int width, int height, int slice, int newSize, uchar* d_radiusMat, uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, int* d_parentMat, uchar* d_statusMat_compact, int* d_childNumMat, short int* d_seedNumberPtr, int* disjointSet_gpu, std::string inputName);