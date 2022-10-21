#pragma once
#include <vector>
#include <iostream>
#include "loadTiff.h"

//Some utils or definations

typedef unsigned char uchar;

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

/*
	loadImage:����sizeָ�룬���潫��������int����ʾwidth,height, slice;
	imagePtrΪ���ص����飬��Ҫ�����ͷ�
*/
uchar* loadImage(const std::string inputName, int* sizes);

void getFiles(std::string path, std::vector<std::string>& files, std::vector<std::string>& names);

enum
{
	FAR, TRIAL, ALIVE, DARKLEAF_PRUNED
};


const int dx[4] = { -1, 0, 0, 1 };
const int dy[4] = { 0, 1, -1, 0 };

const int dx3d[6] = { -1, 1, 0, 0, 0, 0 };
const int dy3d[6] = { 0, 0, -1, 1, 0, 0 };
const int dz3d[6] = { 0, 0, 0, 0, -1, 1 };

const int dx3d26[26] = { -1,-1,-1,-1,-1,-1,-1,-1,-1,   0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1, 1 };
const int dy3d26[26] = { -1,-1,-1, 0, 0, 0, 1, 1, 1,  -1,-1,-1, 0, 0, 1, 1, 1, -1,-1,-1, 0, 0, 0, 1, 1, 1 };
const int dz3d26[26] = { -1, 0, 1,-1, 0, 1,-1, 0, 1,  -1, 0, 1,-1, 1,-1, 0, 1, -1, 0, 1,-1, 0, 1,-1, 0, 1 };

