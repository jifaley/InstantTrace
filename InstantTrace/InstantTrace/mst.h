#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "utils.h"


void primMST(uchar* d_imagePtr_compact, std::vector<int> & seedArr, short int* d_seedNumberPtr, int* d_compress, int* d_decompress, uchar* d_radiusMat_compact, uchar* d_activeMat_compact, int* d_parentPtr_compact, int width, int height, int slice, int newSize);