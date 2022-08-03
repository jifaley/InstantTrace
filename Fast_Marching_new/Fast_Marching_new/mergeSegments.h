#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "utils.h"
#include <map>

void mergeSegments(std::vector<int>& seedArr, std::vector<int>& disjointSet, int width, int height, int slice, int newSize, uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, int* d_childNumMat, uchar* d_radiusMat, int* d_parentPtr, short int* d_seedNumberPtr, int* d_disjointSet);