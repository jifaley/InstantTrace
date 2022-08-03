#pragma once
#include <vector>
#include <iostream>
#include "loadTiff.h"


namespace config
{
	const int __USE__DIST26__CONST = 1;
}



typedef unsigned char uchar;

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

/*
	loadImage:传入size指针，里面将放入三个int，表示width,height, slice;
	imagePtr为返回的数组，需要自行释放
*/
uchar* loadImage(const std::string inputName, int* sizes);

void getFiles(std::string path, std::vector<std::string>& files, std::vector<std::string>& names);

enum
{
	FAR, TRIAL, ALIVE, DARKLEAF_PRUNED
};

enum
{
	FARAWAY, ACTIVE, PASSIVE, Potential
};


const int dx[4] = { -1, 0, 0, 1 };
const int dy[4] = { 0, 1, -1, 0 };

const int dx3d[6] = { -1, 1, 0, 0, 0, 0 };
const int dy3d[6] = { 0, 0, -1, 1, 0, 0 };
const int dz3d[6] = { 0, 0, 0, 0, -1, 1 };

const int dx3d26[26] = { -1,-1,-1,-1,-1,-1,-1,-1,-1,   0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1, 1 };
const int dy3d26[26] = { -1,-1,-1, 0, 0, 0, 1, 1, 1,  -1,-1,-1, 0, 0, 1, 1, 1, -1,-1,-1, 0, 0, 0, 1, 1, 1 };
const int dz3d26[26] = { -1, 0, 1,-1, 0, 1,-1, 0, 1,  -1, 0, 1,-1, 1,-1, 0, 1, -1, 0, 1,-1, 0, 1,-1, 0, 1 };

