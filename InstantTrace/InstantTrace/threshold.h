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
������addGlobalThreshold
���ܣ���d_imagePtr ָ���ͼ�����ȫ����ֵ
*/
void addGlobalThreshold(uchar* d_imagePtr, int width, int height, int slice, uchar threshold);

/*
������addLocalThreshold
���ܣ���d_imagePtr ָ���ͼ����Ӿֲ���ֵ
ʵ�֣����ȸ���blockSize������ͼ��ֿ飬�ֱ�ͳ�ƻҶ�ֱ��ͼ��ֻ����������������ǰ5%��ֵ��
���ݣ���Ԫ�ź�һ����Ա�����˵�������ġ�
ȱ�㣺�������Ϊ���Եķֿ�ЧӦ��Ӧ����Ӳ�ֵ������������
*/
void addLocalThreshold(uchar* d_imagePtr, int width, int height, int slice, int blockSize);


/*
������addDarkPadding
���ܣ���d_imagePtr ָ���ͼ����в���
ʵ�֣������㹻�������򣬽����ܱߵİ����Ҷ���Ϊ1
���ݣ���ͼ���ͬ����֮��ķ�϶��ʹ�ú���׷��ʱ�ܳɹ��������ڵ�����
*/
void addDarkPadding(uchar* d_imagePtr, int width, int height, int slice, uchar threshold);


int getGlobalThreshold(uchar* h_imagePtr, uchar* d_imagePtr, int width, int height, int slice);