#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "utils.h"

#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>


/*
func: equalizeHistogramGpu
输入:inputPtr, width, height, slice
输出:outputPtr, host上的结果
d_output, device上的结果
以上三个数组请提前分配好内存

*/
void equalizeHistogramGpu(uchar* const inputPtr, uchar* const outputPtr, int width, int height, int slice, int *sizes, uchar* d_output);