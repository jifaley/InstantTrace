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
����:inputPtr, width, height, slice
���:outputPtr, host�ϵĽ��
d_output, device�ϵĽ��
����������������ǰ������ڴ�

*/
void equalizeHistogramGpu(uchar* const inputPtr, uchar* const outputPtr, int width, int height, int slice, int *sizes, uchar* d_output);