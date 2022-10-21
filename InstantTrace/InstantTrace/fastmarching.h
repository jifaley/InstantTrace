#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "utils.h"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/scatter.h>
#include <cooperative_groups.h>


#define sq2 (1.414f)
#define sq3 (1.732f)

__constant__ const int dx3dconst[6] = { -1, 1, 0, 0, 0, 0 };
__constant__ const int dy3dconst[6] = { 0, 0, -1, 1, 0, 0 };
__constant__ const int dz3dconst[6] = { 0, 0, 0, 0, -1, 1 };

__constant__ const int dx3d26const[26] = { -1,-1,-1,-1,-1,-1,-1,-1,-1,   0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1, 1 };
__constant__ const int dy3d26const[26] = { -1,-1,-1, 0, 0, 0, 1, 1, 1,  -1,-1,-1, 0, 0, 1, 1, 1, -1,-1,-1, 0, 0, 0, 1, 1, 1 };
__constant__ const int dz3d26const[26] = { -1, 0, 1,-1, 0, 1,-1, 0, 1,  -1, 0, 1,-1, 1,-1, 0, 1, -1, 0, 1,-1, 0, 1,-1, 0, 1 };

__constant__ const float EuclidDistconst[26] = {sq3, sq2, sq3, sq2, 1, sq2, sq3, sq2, sq3, 
                                                sq2,  1,  sq2,  1,      1,  sq2,  1,  sq2,         
												sq3, sq2, sq3, sq2, 1, sq2, sq3, sq2, sq3
};

void addGreyWeightTransform(uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, int width, int height, int slice, int newSize);
void buildInitNeuron(std::vector<int>& seedArr, uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, int* d_parentPtr_compact, short int* d_seedNumberPtr, uchar* d_activeMat_compact, int* d_childNumMat, int width, int height, int slice, int newSize);
void calcRadius_gpu_compact(uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, uchar* d_radiusMat_compact, int width, int height, int slice, int newSize, int globalThreshold);
void calcRadius_gpu_fastmarching(uchar* imagePtr, int* d_compress, int* d_decompress, uchar* d_radiusMat, int width, int height, int slice, int newSize);

void changeSimpleParentToFull(int* d_compress, int* d_decompress, int* d_parentPtr, uchar* d_parentSimplePtr, std::vector<int>& seedArr, int width, int height, int slice, int newSize);