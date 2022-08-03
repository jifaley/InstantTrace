#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "utils.h"
#include <thrust/replace.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

/*
函数：addGlobalThreshold
功能：给d_imagePtr 指向的图像添加全局阈值
*/
void addGlobalThreshold(uchar* d_imagePtr, int width, int height, int slice, uchar threshold);

/*
函数：addLocalThreshold
功能：给d_imagePtr 指向的图像添加局部阈值
实现：首先根据blockSize对整个图像分块，分别统计灰度直方图。只保留块内亮度排名前5%的值。
根据：神经元信号一般相对背景来说是明亮的。
缺点：会产生较为明显的分块效应，应该添加插值等修正方法。
*/
void addLocalThreshold(uchar* d_imagePtr, int width, int height, int slice, int blockSize);


/*
函数：addDarkPadding
功能：给d_imagePtr 指向的图像进行补充
实现：对于足够亮的区域，将其周边的暗区灰度置为1
根据：试图填补不同亮区之间的缝隙，使得后面追踪时能成功连接相邻的亮区
*/
void addDarkPadding(uchar* d_imagePtr, int width, int height, int slice, uchar threshold);


int getGlobalThreshold(uchar* h_imagePtr, uchar* d_imagePtr, int width, int height, int slice);