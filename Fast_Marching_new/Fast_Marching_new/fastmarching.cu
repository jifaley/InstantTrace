#include "fastmarching.h"
//#define __ALLOW__GAP
#define __USE__DIST26

#include "TimerClock.hpp"

using namespace cooperative_groups;

//#define __NON__ATOMIC

//static texture<float> inputTexture;
__constant__ const int smallqueueSize = 10;
__constant__ const int smallqueueNumber = 26177;



__device__ float fatomicMin(float *addr, float value) {
	float old = *addr, assumed;
	if (old <= value) return old;
	do {
		assumed = old;
		old = atomicCAS((int*)addr, __float_as_int(assumed), __float_as_int(MIN(value, assumed)));
	} while (old != assumed);
	//printf("%f %f\n",MIN(value,assumed), old);
	return old;
};


//函数：gwdtExtendKernel
//功能:通过最短路的方法计算GreyWeight Distance Transform。本Kernel用于将某个像素向邻居扩展。
__global__
void gwdtExtendKernel(uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_frontier_compact, int* d_compress, int* d_decompress, int* d_distPtr, int* d_updateDistPtr, uchar* d_inCurFrontier, uchar* d_inNextFrontier,  int width, int height, int slice, int newSize, int compact_size)
{
	//smallIdx: 压缩后的下标 fullIdx: 原始图像的下标 newSize: 压缩后图像的大小
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= compact_size) return;

	int smallIdx = d_frontier_compact[tid];
	{
		d_inCurFrontier[smallIdx] = 0;
		if (smallIdx >= newSize) return;

		//判断该点是否刚被更新过

		int fullIdx = d_decompress[smallIdx];

		int3 curPos;
		curPos.z = fullIdx / (width * height);
		curPos.y = fullIdx % (width * height) / width;
		curPos.x = fullIdx % width;

		int3 neighborPos;
		int neighborIdx, neighborSmallIdx;
		int neighborValue;

		int curDist = d_distPtr[smallIdx];

		for (int k = 0; k < 6; k++)
		{
			neighborPos.x = curPos.x + dx3dconst[k];
			neighborPos.y = curPos.y + dy3dconst[k];
			neighborPos.z = curPos.z + dz3dconst[k];
			if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
				|| neighborPos.z < 0 || neighborPos.z >= slice)
				continue;
			neighborIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;
			neighborValue = d_imagePtr[neighborIdx];

			//若邻居为背景则不扩展
			if (neighborValue == 0) continue;

			//最短路计算方法：从所有背景像素出发，边的长度为像素值
			neighborSmallIdx = d_compress[neighborIdx];
			int old = atomicMin(&d_updateDistPtr[neighborSmallIdx], curDist + neighborValue);

			//old返回的是原子操作之前的值。如果发现updateDist数组被更新了，激活该点。
			if (curDist + neighborValue < old)
				d_inNextFrontier[neighborSmallIdx] = 1;
				//d_nextStatus[neighborSmallIdx] = ACTIVE;
		}
	}
}


//函数：gwdtUpdateKernel
//功能:通过最短路的方法计算GreyWeight Distance Transform。本Kernel用于更新节点的dist值。
__global__
void gwdtUpdateKernel(uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_frontier_compact, int* d_distPtr, int* d_updateDistPtr, uchar* d_inCurFrontier, uchar* d_inNextFrontier,  int width, int height, int slice, int newSize, int compact_size)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= compact_size) return;
	int smallIdx = d_frontier_compact[tid];

	if (!d_inNextFrontier[smallIdx]) return;
	d_inNextFrontier[smallIdx] = 0;
	//更新阶段到下一个锋面。本像素原来是上一锋面的ExtendKernel()扩展出来的。
	d_inCurFrontier[smallIdx] = 1;

	int updateValue = d_updateDistPtr[smallIdx];
	int curValue = d_distPtr[smallIdx];

	if (updateValue < curValue)
	{
		d_distPtr[smallIdx] = updateValue;
	}
}


//预处理：把所有临近0的像素作为起始点，他们距离背景的距离即为他们的像素值
__global__ void gwdtPreProcessKernel(uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress,  int* d_distPtr, int* d_updateDistPtr,  uchar* d_inCurFrontier, int width, int height, int slice, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;
	int3 neighborPos;
	int3 curPos;
	uchar neighborValue;
	uchar curValue;

	curValue = d_imagePtr_compact[smallIdx];
	int fullIdx = d_decompress[smallIdx];
	int neighborIdx;

	curPos.z = fullIdx / (width * height);
	curPos.y = fullIdx % (width * height) / width;
	curPos.x = fullIdx % width;

	
	for (int k = 0; k < 6; k++)
	{
		neighborPos.x = curPos.x + dx3dconst[k];
		neighborPos.y = curPos.y + dy3dconst[k];
		neighborPos.z = curPos.z + dz3dconst[k];
		if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
			|| neighborPos.z < 0 || neighborPos.z >= slice)
			continue;
		neighborIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;

		neighborValue = d_imagePtr[neighborIdx];

		if (neighborValue == 0)
		{
			d_distPtr[smallIdx] = curValue;
			d_updateDistPtr[smallIdx] = curValue;
			d_inCurFrontier[smallIdx] = 1;
			break;
		}
	}
}


//后处理：将距离变换的结果缩放到0-255（是否应该是1-255？测试发现会让点数急剧增加）
__global__ void GWDT_PostProcess_compact(uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, int* d_distPtr, float maxValue, int width, int height, int slice, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;
	float temp = d_distPtr[smallIdx];
	int fullIdx = d_decompress[smallIdx];
	d_imagePtr_compact[smallIdx] = temp / maxValue * 255;
	d_imagePtr[fullIdx] = temp / maxValue * 255;
}


/*
函数：addGreyWeightTransform
功能：通过最短路迭代，找到每个像素到离他最近的背景像素的距离（点之间距离即为像素值），
然后根据这个距离对原图进行映射。这样映射之后，距离背景更远的像素会更亮（也更有可能是神经纤维的中心），
处理过后，下一步的追踪更容易沿着神经纤维中心拓展。
输出：d_imagePtr （直接在原图上进行改动）
*/

void addGreyWeightTransform(uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, int width, int height, int slice, int newSize)
{
	int* d_distPtr;
	int* d_updateDistPtr;

	std::cerr << "NewSize:" << newSize << std::endl;

	cudaMalloc((void**)&d_distPtr, sizeof(int) * newSize);
	cudaMalloc((void**)&d_updateDistPtr, sizeof(int) * newSize);

	thrust::device_ptr<int> d_distPtr_thrust(d_distPtr);
	thrust::device_ptr<int> d_updateDistPtr_thrust(d_updateDistPtr);
	thrust::fill(d_distPtr_thrust, d_distPtr_thrust + newSize, 100000000);
	thrust::fill(d_updateDistPtr_thrust, d_updateDistPtr_thrust + newSize, 100000000);

	//thrust::fill(d_curStatus_thrust, d_curStatus_thrust + width * height * slice, FARAWAY);

	uchar* d_inCurFrontier;
	uchar* d_inNextFrontier;

	cudaMalloc((void**)&d_inNextFrontier, sizeof(uchar) * newSize);
	cudaMalloc((void**)&d_inCurFrontier, sizeof(uchar) * newSize);

	cudaMemset(d_inNextFrontier, 0, sizeof(uchar) * newSize);
	cudaMemset(d_inCurFrontier, 0, sizeof(uchar) * newSize);

	thrust::device_vector<int>dv_frontier_compact(newSize);
	int* d_frontier_compact = thrust::raw_pointer_cast(dv_frontier_compact.data());


	cudaError_t errorCheck;

	gwdtPreProcessKernel << < (newSize - 1) / 256 + 1, 256 >> > (d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_distPtr, d_updateDistPtr, d_inCurFrontier, width, height, slice, newSize);
	
	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "In GWDT Preprocess: " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	using thrust::placeholders::_1;

	int* d_copy_end;
	int compact_size;



	try
	{
		d_copy_end = thrust::copy_if(thrust::device, thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(newSize), d_inCurFrontier, d_frontier_compact, _1 != 0);
		compact_size = d_copy_end - d_frontier_compact;
	}
	catch (thrust::system_error error)
	{
		std::cerr << std::string(error.what()) << std::endl;
	}


	//blockSize:64
	//maxBlockNum:512
	int counter = 0;
	while (1)
	{
		counter++;
		gwdtExtendKernel << <(newSize - 1) / 256 + 1, 256 >> > (d_imagePtr, d_imagePtr_compact, d_frontier_compact,  d_compress, d_decompress,  d_distPtr, d_updateDistPtr, d_inCurFrontier, d_inNextFrontier,

			width, height, slice, newSize, compact_size);


		d_copy_end = thrust::copy_if(thrust::device, thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(newSize), d_inNextFrontier, d_frontier_compact, _1 != 0);
		compact_size = d_copy_end - d_frontier_compact;
		if (compact_size == 0)
			break;

		gwdtUpdateKernel << <(newSize - 1) / 256 + 1, 256 >> > (d_imagePtr, d_imagePtr_compact, d_frontier_compact, d_distPtr, d_updateDistPtr, d_inCurFrontier, d_inNextFrontier,

			width, height, slice, newSize, compact_size);
	}


	int maxValue = thrust::reduce(d_distPtr_thrust, d_distPtr_thrust + newSize, 0, thrust::maximum<int>());
	std::cerr << "Max value by reduce: " << maxValue << std::endl;

	GWDT_PostProcess_compact << <(newSize - 1) / 256 + 1, 256 >> > (d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_distPtr, maxValue, width, height, slice, newSize);
	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "In GWDT PostProcess: " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	cudaFree(d_distPtr);
	cudaFree(d_updateDistPtr);
}



void findMaxPos(uchar* imagePtr, int width, int height, int slice, int& maxpos)
{
	float maxvalue = -1;
	maxpos = 0;


	for (int i = 0; i < width * height * slice; i++)
	{
		if (imagePtr[i] > imagePtr[maxpos])
			maxpos = i;
	}
	
	maxvalue = imagePtr[maxpos];
	std::cerr << "Max value by GPU " << maxvalue << std::endl;

	//Modified by jifal3y 20211123 maxPos改为各种中心的平均位置

	/*double xSum, ySum, zSum;
	xSum = ySum = zSum = 0;
	double radiusSum = 0;

	int count = 0;
	for (int i = 0; i < width * height * slice; i++)
	{
		if (distMat[i] >= MIN(maxvalue, 10))
		{
			xSum += (i % width);
			ySum += (i % (width * height) / width);
			zSum += (i / (width * height));
			count++;
		}
	}

	int meanX = xSum / count;
	int meanY = ySum / count;
	int meanZ = zSum / count;
	maxpos = meanZ * width * height + meanY * width + meanX;*/
}





//10_26177:2.2s
//32,8237:1.9s


__device__
float gIFunc_gpu(float value, float maxvalue = 255, float lambda = 10)
{
	return exp(lambda * (1 - value * 1.0f / maxvalue) * (1 - value * 1.0f / maxvalue));
	//gI(x) = exp(\lambda_i * (1 - I(x)/Imax) ** 2)
}

__device__
float gwdtFunc_gpu(float dist, float value1, float value2)
{
	return dist * (gIFunc_gpu(value1) + gIFunc_gpu(value2)) / 2.f;
	//e(x,y) = |x-y| * (gI(x) + gI(y)) / 2
}

//流压缩后的数组长度
__device__ int d_compact_size;


/*
函数:tracingExtendKernel
功能:通过最短路的方法寻找神经元走向。本Kernel用于将某个像素向邻居扩展。
原理:经过我们之前的灰度距离映射(GWDT)，现在的最短路会更倾向于向神经纤维的中心扩展(越亮,dist越小)。
*/
__global__
void tracingExtendKernel(uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_frontier_compact, int* d_compress, int* d_decompress, float* d_distPtr, float* d_updateDistPtr, int* d_inCurFrontier, int* d_inNextFrontier,
	int width, int height, int slice, int newSize, int compact_size)
{
	int start = threadIdx.x + blockIdx.x*blockDim.x;
	if (start >= compact_size) return;
	int tid;

	for (tid = start; tid < compact_size; tid += blockDim.x * gridDim.x)
	{
		//每次从frontier里面获取一个点,里面存放的是流压缩后的下标
		int smallIdx = d_frontier_compact[tid];
		{
			d_inCurFrontier[smallIdx] = 0;

			//curValue: 当前点的像素值
			uchar curValue = d_imagePtr_compact[smallIdx];
			//if (curValue == 0) return;

			//curDist: 当前点的最短路dist值
			float curDist = d_distPtr[smallIdx];
			//解压缩，回到压缩之前的下标
			int fullIdx = d_decompress[smallIdx];

			int3 curPos;
			curPos.z = fullIdx / (width * height);
			curPos.y = fullIdx % (width * height) / width;
			curPos.x = fullIdx % width;

			int3 neighborPos;
			int neighborIdx;
			int neighborSmallIdx;
			uchar neighborValue;

			//向周围26个方向扩展
			for (int k = 0; k < 26; k++)
			{
				neighborPos.x = curPos.x + dx3d26const[k];
				neighborPos.y = curPos.y + dy3d26const[k];
				neighborPos.z = curPos.z + dz3d26const[k];

				if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
					|| neighborPos.z < 0 || neighborPos.z >= slice)
					continue;
				neighborIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;

				neighborValue = d_imagePtr[neighborIdx];
				if (neighborValue == 0) continue;

				neighborSmallIdx = d_compress[neighborIdx];
				float EuclidDist = 1;
				//EuclidDist = sqrtf(dx3d26const[k] * dx3d26const[k] + dy3d26const[k] * dy3d26const[k] + dz3d26const[k] * dz3d26const[k]);
				//由于只有26个邻居，直接把对应的欧式距离存储起来了
				EuclidDist = EuclidDistconst[k];
				//两点之间的dist根据两点的欧式距离和亮度计算
				float deltaDist = gwdtFunc_gpu(EuclidDist, curValue, neighborValue);

#ifdef __NON__ATOMIC
				float newDist = curDist + deltaDist;
				newDist = __int_as_float(__float_as_int(newDist) & 0xFFFFFF00 | k);

				if (d_updateDistPtr[neighborSmallIdx] - 1e-5 > newDist)
				{
					d_updateDistPtr[neighborSmallIdx] = newDist;
					d_inNextFrontier[neighborSmallIdx] = 1;
				}
#else

				//fastcheck
				//在使用原子操作之前，进行一次快速检查。如果当前点连上阶段的邻居都更新不了，就放弃更新
				if (d_distPtr[neighborSmallIdx] - 1e-5 < curDist + deltaDist)
					continue;

				float newDist = curDist + deltaDist;
				//在dist的后面8个bit放入更新使用的方向k
				newDist = __int_as_float(__float_as_int(newDist) & 0xFFFFFF00 | k);

				//oldDist是atomicMin()返回的值，返回的是此次原子修改前的值,无论是否成功
				int oldDist = atomicMin((int*)(d_updateDistPtr + neighborSmallIdx), __float_as_int(newDist));
				//如果修改成功了
				if (__int_as_float(oldDist) > newDist)
					d_inNextFrontier[neighborSmallIdx] = 1;
#endif
			}

		}
	}
}




__global__
void tracingExtendKernel_warpShuffle(uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_frontier_compact, int* d_compress, int* d_decompress, float* d_distPtr, float* d_updateDistPtr, int* d_inCurFrontier, int* d_inNextFrontier,
	int width, int height, int slice, int newSize, int compact_size)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int warp_id = threadIdx.x / 32;
	int lane_id = threadIdx.x % 32;
	int pointId = blockIdx.x * blockDim.x / 32 + warp_id;

	if (pointId >= compact_size) return;

	int smallIdx, fullIdx;
	int curValue;
	float curDist;

	auto g = coalesced_threads();
	
	//if (g.thread_rank() == 0)
	//{
	//	smallIdx = d_frontier_compact[pointId];
	//	d_inCurFrontier[smallIdx] = 0;
	//	curValue = d_imagePtr_compact[smallIdx];
	//	curDist = d_distPtr[smallIdx];
	//	fullIdx = d_decompress[smallIdx];
	//}
	//g.shfl(fullIdx, 0);
	//g.shfl(curDist, 0);
	//g.shfl(curValue, 0);

	if (lane_id == 0)
	{
		smallIdx = d_frontier_compact[pointId];
		d_inCurFrontier[smallIdx] = 0;
		curValue = d_imagePtr_compact[smallIdx];
		curDist = d_distPtr[smallIdx];
		fullIdx = d_decompress[smallIdx];
	}

	fullIdx = __shfl_sync(-1, fullIdx, 0);
	curDist = __shfl_sync(-1, curDist, 0);
	curValue = __shfl_sync(-1, curValue, 0);

	int3 curPos;
	curPos.z = fullIdx / (width * height);
	curPos.y = fullIdx % (width * height) / width;
	curPos.x = fullIdx % width;

	int3 neighborPos;
	int neighborIdx;
	int neighborSmallIdx;
	uchar neighborValue;

	int k = lane_id;

	if (k < 26)
	{
		neighborPos.x = curPos.x + dx3d26const[k];
		neighborPos.y = curPos.y + dy3d26const[k];
		neighborPos.z = curPos.z + dz3d26const[k];

		if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
			|| neighborPos.z < 0 || neighborPos.z >= slice)
			return;
		neighborIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;

		neighborValue = d_imagePtr[neighborIdx];
		if (neighborValue == 0) return;

		neighborSmallIdx = d_compress[neighborIdx];
		float EuclidDist = 1;
		//EuclidDist = sqrtf(dx3d26const[k] * dx3d26const[k] + dy3d26const[k] * dy3d26const[k] + dz3d26const[k] * dz3d26const[k]);
		//由于只有26个邻居，直接把对应的欧式距离存储起来了
		EuclidDist = EuclidDistconst[k];
		//两点之间的dist根据两点的欧式距离和亮度计算
		float deltaDist = gwdtFunc_gpu(EuclidDist, curValue, neighborValue);
		//fastcheck

		//在使用原子操作之前，进行一次快速检查。如果当前点连上阶段的邻居都更新不了，就放弃更新
		if (d_distPtr[neighborSmallIdx] - 1e-5 < curDist + deltaDist)
			return;

		float newDist = curDist + deltaDist;
		//在dist的后面8个bit放入更新使用的方向k
		newDist = __int_as_float(__float_as_int(newDist) & 0xFFFFFF00 | k);

		//oldDist是atomicMin()返回的值，返回的是此次原子修改前的值,无论是否成功
		int oldDist = atomicMin((int*)(d_updateDistPtr + neighborSmallIdx), __float_as_int(newDist));

		//float oldDist = d_updateDistPtr[neighborSmallIdx];

		//如果修改成功了
		if (__int_as_float(oldDist) > newDist)
		//if (oldDist > newDist)
		{
			//d_updateDistPtr[neighborSmallIdx] = newDist;
			d_inNextFrontier[neighborSmallIdx] = 1;
		}
	}
}

__device__ int atomicAggInc_coop() {
	auto g = coalesced_threads();
	int warp_res;
	if (g.thread_rank() == 0)
		warp_res = atomicAdd(&d_compact_size, g.size());
	return g.shfl(warp_res, 0) + g.thread_rank();
}

__global__ void filter_k(int *dst, const int *src, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n)
		return;
	if (src[i] > 0)
		dst[atomicAggInc_coop()] = i;
}



__global__
void tracingExtendKernel_warpShuffle_atomic(uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_frontier_compact, int* d_compress, int* d_decompress, float* d_distPtr, float* d_updateDistPtr,
	int width, int height, int slice, int newSize, int compact_size, int* d_frontier_compact_2)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	__shared__ int blockLength;
	__shared__ int blockOffset;

	int warp_id = threadIdx.x / 32;
	int lane_id = threadIdx.x % 32;
	int pointId = blockIdx.x * blockDim.x / 32 + warp_id;

	if (pointId >= compact_size) return;

	int smallIdx, fullIdx;
	int curValue;
	float curDist;

	//auto g = coalesced_threads();

	//if (g.thread_rank() == 0)
	//{
	//	smallIdx = d_frontier_compact[pointId];
	//	curValue = d_imagePtr_compact[smallIdx];
	//	curDist = d_distPtr[smallIdx];
	//	fullIdx = d_decompress[smallIdx];
	//}
	//fullIdx = g.shfl(fullIdx, 0);
	//curDist = g.shfl(curDist, 0);
	//curValue = g.shfl(curValue, 0);

	if (lane_id == 0)
	{
		smallIdx = d_frontier_compact[pointId];
		curValue = d_imagePtr_compact[smallIdx];
		curDist = d_distPtr[smallIdx];
		fullIdx = d_decompress[smallIdx];
	}

	fullIdx = __shfl_sync(-1, fullIdx, 0);
	curDist = __shfl_sync(-1, curDist, 0);
	curValue = __shfl_sync(-1, curValue, 0);

	int3 curPos;
	curPos.z = fullIdx / (width * height);
	curPos.y = fullIdx % (width * height) / width;
	curPos.x = fullIdx % width;

	int3 neighborPos;
	int neighborIdx;
	int neighborSmallIdx;
	uchar neighborValue;

	int k = lane_id;
	int modified = 0;

	if (k < 26)
	{
		neighborPos.x = curPos.x + dx3d26const[k];
		neighborPos.y = curPos.y + dy3d26const[k];
		neighborPos.z = curPos.z + dz3d26const[k];

		if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
			|| neighborPos.z < 0 || neighborPos.z >= slice)
			return;
		neighborIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;

		neighborValue = d_imagePtr[neighborIdx];
		if (neighborValue == 0) return;

		neighborSmallIdx = d_compress[neighborIdx];
		float EuclidDist = 1;
		//EuclidDist = sqrtf(dx3d26const[k] * dx3d26const[k] + dy3d26const[k] * dy3d26const[k] + dz3d26const[k] * dz3d26const[k]);
		//由于只有26个邻居，直接把对应的欧式距离存储起来了
		EuclidDist = EuclidDistconst[k];
		//两点之间的dist根据两点的欧式距离和亮度计算
		float deltaDist = gwdtFunc_gpu(EuclidDist, curValue, neighborValue);
		//fastcheck

		//在使用原子操作之前，进行一次快速检查。如果当前点连上阶段的邻居都更新不了，就放弃更新
		if (d_distPtr[neighborSmallIdx] - 1e-5 < curDist + deltaDist)
			return;

		float newDist = curDist + deltaDist;
		//在dist的后面8个bit放入更新使用的方向k
		newDist = __int_as_float(__float_as_int(newDist) & 0xFFFFFF00 | k);

		//oldDist是atomicMin()返回的值，返回的是此次原子修改前的值,无论是否成功
		int oldDist = atomicMin((int*)(d_updateDistPtr + neighborSmallIdx), __float_as_int(newDist));

		//如果修改成功了
		if (__int_as_float(oldDist) > newDist)
		{
			modified = 1;
		}
	}

	int warpOffset;

	if (modified)
	{
		//int pos = atomicAdd(&d_compact_size, 1);
		//d_frontier_compact_2[pos] = neighborSmallIdx;

		auto g = coalesced_threads();
		int warp_res;
		int rank = g.thread_rank();
		if (rank == 0)
			warp_res = atomicAdd(&d_compact_size, g.size());

		warp_res = g.shfl(warp_res, 0);
		int result =  warp_res + rank;

		d_frontier_compact_2[result] = neighborSmallIdx;
	}
}


/*
函数:tracingUpdateKernel
功能:通过最短路的方法寻找神经元走向。本Kernel用于更新某个像素。
*/
__global__
void tracingUpdateKernel(int* d_compress, int* d_decompress, int* d_frontier_compact, float* d_distPtr, float* d_updateDistPtr, uchar* parentSimplePtr_compact, uchar* d_activeMat_compact, int* d_childNumMat_INT, short int* d_seedNumberPtr,
	int width, int height, int slice, int newSize, int compact_size)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid == 0)
		d_compact_size = 0;
	if (tid >= compact_size) return;
	int smallIdx = d_frontier_compact[tid];
	
	int fullIdx = d_decompress[smallIdx];
	//更新之前的direction
	int direction = parentSimplePtr_compact[smallIdx];
	//更新之前的seed(就是属于哪个种子)
	int curSeed = d_seedNumberPtr[smallIdx];
	int z = fullIdx / (width * height);
	int y = fullIdx % (width * height) / width;
	int x = fullIdx % width;

	//更新之后(存储再updateDist中的) dist和direction
	float newDist = d_updateDistPtr[smallIdx];
	int directionUpdate = __float_as_int(newDist) & 0xFF;

	//更新之后的parent(根据directionUpdate计算)
	int newParent;
	if (directionUpdate == 0xff)
		newParent = -1;
	else
		newParent = (z - dz3d26const[directionUpdate]) * width * height + (y - dy3d26const[directionUpdate]) * width + (x - dx3d26const[directionUpdate]);
	int newParentSmallIdx = d_compress[newParent];

	//更新之后的seed
	//注意：这里用的是当前的parent的seed。之后，这个parent的seed可能还会发生改变，从而发生mergesegments里面的
	//rb != rc的情况。
	int newSeed = d_seedNumberPtr[newParentSmallIdx];

	//如果更新前后seed不相同，说明该点更新前后属于两个不同分支。
	//让新的parent作为Top-1，旧的parent作为Top-2保存下来。
	if (curSeed != newSeed)
	{
		d_seedNumberPtr[smallIdx] = newSeed;
		//Top-2的parent放在Top-1数组后面
		parentSimplePtr_compact[smallIdx + newSize] = direction;
	}

	d_distPtr[smallIdx] = newDist;
	parentSimplePtr_compact[smallIdx] = directionUpdate;
	d_activeMat_compact[smallIdx] = ALIVE;
}

__global__
void tracingPreprocessKernel(uchar* d_imagePtr, int* d_compress, int* seedArr, float* d_distPtr, float* d_updateDistPtr, uchar* d_activeMat_compact, int* d_frontier_compact, short int* d_seedNumberPtr, int width, int height, int slice, int seedCount)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= seedCount) return;
	int fullIdx = seedArr[idx];
	int smallIdx = d_compress[fullIdx];

	d_distPtr[smallIdx] = 0;
	d_updateDistPtr[smallIdx] = 0;
	//parentPtr[index] = index;
	d_activeMat_compact[smallIdx] = ALIVE;
	d_frontier_compact[idx] = smallIdx;
	//Adding Pruning Merge 20211030
	//因为初始值为0，所以改了
	d_seedNumberPtr[smallIdx] = idx + 1;
}


struct transform_functor
{
	__host__ __device__
	int operator()(const uchar& flag, int& prefixSum) const
	{
		return flag == 1 ? prefixSum : 0;
	}
};

struct compact_functor
{
	__host__ __device__
	bool operator()(const uchar & x)
	{
		return x != 0;
	}
};

__global__
void getCompressMap3(int* d_frontier_compact, int* d_prefixSum, uchar* d_flag, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;
	if (d_flag[smallIdx] == 1)
	{
		int smallIdx_L2 = d_prefixSum[smallIdx];
		d_frontier_compact[smallIdx_L2] = smallIdx;
	}
}


__global__
void buildInitNeuron_Kernel_Parent(uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, int* d_parentPtr_compact, short int* d_seedNumberPtr, uchar* d_activeMat_compact, int* d_childNumMat_INT, float* d_distPtr, float* d_updateDistPtr, uchar* d_parentSimplePtr,
	int* d_frontier_compact, int* d_prefixSum, int width, int height, int slice, int newSize, int compact_size_init, int* d_frontier_compact2)
{
	int counter = 0;

	int last_compact_size = compact_size_init;
	int blockNum = (newSize - 1) / 128 + 1;

	int ping_pong = 0;

	int* f1;
	int* f2;

	while (1)
	{
		//std::cerr << counter << std::endl;

		if (ping_pong == 0)
		{
			f1 = d_frontier_compact; f2 = d_frontier_compact2;
		}
		else
		{
			f1 = d_frontier_compact2; f2 = d_frontier_compact;
		}


		tracingExtendKernel_warpShuffle_atomic << <(last_compact_size -1)/32 + 1, 1024 >> > (d_imagePtr, d_imagePtr_compact, f1, d_compress, d_decompress, d_distPtr, d_updateDistPtr, 
			width, height, slice, newSize, last_compact_size, f2);
	
		//filter_k << < blockNum, 128 >> > (d_frontier_compact, d_inNextFrontier, newSize);

		cudaDeviceSynchronize();
		last_compact_size = atomicAdd(&d_compact_size, 0);
		d_compact_size = 0;


		if (last_compact_size == 0)
			break;

		tracingUpdateKernel << <(last_compact_size - 1) / 256 + 1, 256 >> > (d_compress, d_decompress, f2, d_distPtr, d_updateDistPtr, d_parentSimplePtr, d_activeMat_compact, d_childNumMat_INT, d_seedNumberPtr,
			width, height, slice, newSize, last_compact_size);
		counter++;

		ping_pong = 1 - ping_pong;
	}
}


/*
函数：buildInitNeuron
功能：从若干个种子开始,通过最短路迭代，以最短路访问顺序作为初始追踪结果
参数: seedArr(种子集合)
d_imagePtr(原图)， d_imagePtr_compact(压缩后原图), d_compress, d_decompress(压缩使用的映射)，
width, height, slice, newSize(压缩后的数组大小)，
输出：d_parentPtr(每个节点的parent信息,实际就是追踪的结果)
*/
void buildInitNeuron(std::vector<int>& seedArr, uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, int* d_parentPtr_compact, short int* d_seedNumberPtr, uchar* d_activeMat_compact, int* d_childNumMat_INT, int width, int height, int slice, int newSize)
{
	cudaError_t errorCheck;

	//d_distPtr: 用于存储最短路的dist值
	//d_updateDistPtr: 用于存储即将更新的dist值(并行最短路中，需要先存储将要更新的dist，同步后再更新
	float* d_distPtr;
	float* d_updateDistPtr;

	cudaMalloc((void**)&d_distPtr, sizeof(float) * newSize);
	cudaMalloc((void**)&d_updateDistPtr, sizeof(float) * newSize);

	thrust::fill(thrust::device, d_distPtr, d_distPtr + newSize, 1e10f);
	thrust::fill(thrust::device, d_updateDistPtr, d_updateDistPtr + newSize, 1e10f);

	//d_parentSimplePtr: 为了让parent信息能和dist信息存储在同一个float32中，将parent信息压缩为一个字节(即，只存储一个方向)
	uchar* d_parentSimplePtr;
	cudaMalloc((void**)& d_parentSimplePtr, sizeof(uchar) * newSize * 2);
	cudaMemset(d_parentSimplePtr, 0xff, sizeof(uchar) * newSize * 2);

	//d_frontier_compact: 存储对锋面压缩后的结果，仅存储在锋面中的元素的下标
	thrust::device_vector<int>dv_frontier_compact(newSize);
	int* d_frontier_compact = thrust::raw_pointer_cast(dv_frontier_compact.data());

	thrust::device_vector<int>dv_frontier_compact2(newSize);
	int* d_frontier_compact2 = thrust::raw_pointer_cast(dv_frontier_compact2.data());
	
	int seedCount = seedArr.size();
	int* d_seedArr;
	cudaMalloc((void**)&d_seedArr, sizeof(int) * seedCount);
	cudaMemcpy(d_seedArr, &seedArr[0], sizeof(int) * seedCount, cudaMemcpyHostToDevice);


	tracingPreprocessKernel << <(seedCount - 1) / 256 + 1, 256 >> > (d_imagePtr, d_compress, d_seedArr, d_distPtr, d_updateDistPtr, d_activeMat_compact, d_frontier_compact, d_seedNumberPtr, width, height, slice, seedCount);

	int compact_size;

	compact_size = seedCount;



	int counter = 0;
	int blockNum = (newSize - 1) / 512 + 1;

	int* h_result;
	int* d_result;
	cudaHostAlloc(&h_result, sizeof(int), cudaHostRegisterMapped);
	*h_result = 0;
	cudaHostGetDevicePointer(&d_result, h_result, 0);

	int ping_pong = 0;
	int* f1;
	int* f2;

	while (1)
	{
		//std::cerr << counter << std::endl;

		if (ping_pong == 0)
		{
			f1 = d_frontier_compact; f2 = d_frontier_compact2;
		}
		else
		{
			f1 = d_frontier_compact2; f2 = d_frontier_compact;
		}

		tracingExtendKernel_warpShuffle_atomic << <(compact_size - 1) / 32 + 1, 1024 >> > (d_imagePtr, d_imagePtr_compact, f1, d_compress, d_decompress, d_distPtr, d_updateDistPtr,
			width, height, slice, newSize, compact_size, f2);

		cudaMemcpyFromSymbol((void*)&compact_size, d_compact_size, sizeof(int));

		if (compact_size == 0)
			break;

		tracingUpdateKernel << <(compact_size-1)/256+1, 256 >> > (d_compress, d_decompress, f2,  d_distPtr, d_updateDistPtr, d_parentSimplePtr,  d_activeMat_compact, d_childNumMat_INT, d_seedNumberPtr,
			width, height, slice, newSize, compact_size);

		ping_pong = 1 - ping_pong;
		counter++;
	}

	

	//buildInitNeuron_Kernel_Parent << <1, 1 >> > (d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_parentPtr_compact, d_seedNumberPtr, d_activeMat_compact, d_childNumMat_INT, d_distPtr, d_updateDistPtr, d_parentSimplePtr, 
	//	d_frontier_compact, d_prefixSum, width, height, slice, newSize, compact_size, d_frontier_compact2);

	//初始追踪结束以后，将存储parent的方向改为存储完整的parent信息用于后面的处理
	changeSimpleParentToFull(d_compress, d_decompress, d_parentPtr_compact, d_parentSimplePtr, seedArr, width, height, slice, newSize);

	cudaFree(d_distPtr);
	cudaFree(d_updateDistPtr);
	cudaFree(d_seedArr);
}


/*
函数:calcRadiusKernel
功能：计算出每个点的控制半径,以便后面剪枝的时候判断覆盖区域
*/

__global__ void calcRadiusKernel_compact(uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, uchar* d_radiusMat_compact, int width, int height, int slice, int newSize, int globalThreshold)
{
	int smallIdx = threadIdx.x + blockDim.x * blockIdx.x;
	if (smallIdx >= newSize) return;
	//Modified by jifaley 20210928
	//change from 15 to 5
	int darkElementThreshold = 15;
	darkElementThreshold = 5;
	int calcThreshold = 5;
	calcThreshold = 2;

	darkElementThreshold = globalThreshold;
	calcThreshold = globalThreshold;

	if (d_imagePtr_compact[smallIdx] < calcThreshold) return;
	float coverRatio = 0.5;

	//Modified by jifaley 20211123 为了对胞体处理
	coverRatio = 0.7;
	//0.9是为了得到125的结果
	coverRatio = 0.9;

	coverRatio = 0.001; //for APP2

	//coverRatio = 0.5;


	coverRatio = 0.1;

	//coverRatio = 0.9;

	int fullIdx = d_decompress[smallIdx];

	int s = fullIdx / (width * height);
	int i = (fullIdx % (width * height)) / width;
	int j = fullIdx % width;

	int r = 1;
	int rmax = 40;

	rmax = MIN(rmax, s);
	rmax = MIN(rmax, slice - s - 1);
	rmax = MIN(rmax, i);
	rmax = MIN(rmax, height - i - 1);
	rmax = MIN(rmax, j);
	rmax = MIN(rmax, width - j - 1);


	int totalPixel = 0;
	int totalDarkPixel = 0;
	int x, y, z, tempIndex;
	while (r < rmax)
	{
		//考虑一下，是否要添加这个限制？20220116
	/*	if (s - r < 0 || s + r >= slice || i - r < 0 || i + r >= height || j - r < 0 || j + r >= width)
		{
			while (r > 0 && s - r < 0 || s + r >= slice || i - r < 0 || i + r >= height || j - r < 0 || j + r >= width)
				r--;
			break;
		}*/
		if (r == 1)
		{
			for (z = MAX(s - r, 0); z <= MIN(s + r, slice - 1); z++)
				for (y = MAX(i - r, 0); y <= MIN(i + r, height - 1); y++)
					for (x = MAX(j - r, 0); x <= MIN(j + r, width - 1); x++)
					{
						tempIndex = z * width * height + y * width + x;
						totalPixel++;
						if (d_imagePtr[tempIndex] < darkElementThreshold)
							totalDarkPixel++;
					}
		}
		else
		{
			//下面是正方体覆盖
			//face +-z

			for (z = s - r; z <= s + r; z += 2 * r)
				if (z >= 0 && z < slice)
				{
					for (y = MAX(i - r, 0); y <= MIN(i + r, height - 1); y++)
						for (x = MAX(j - r, 0); x <= MIN(j + r, width - 1); x++)
						{

							float dist = sqrtf((z - s) * (z - s) + (y - i) * (y - i) + (x - j) * (x - j));
							if (dist > r - 1 && dist <= r)
							{
								totalPixel++;
								tempIndex = z * width * height + y * width + x;
								if (d_imagePtr[tempIndex] < darkElementThreshold)
									totalDarkPixel++;
							}
						}
				}
			////face +-y
			for (y = i - r; y <= i + r; y += 2 * r)
				if (y >= 0 && y < height)
				{
					for (z = MAX(s - r, 0); z <= MIN(s + r, slice - 1); z++)
						for (x = MAX(j - r, 0); x <= MIN(j + r, width - 1); x++)
						{
							float dist = sqrtf((z - s) * (z - s) + (y - i) * (y - i) + (x - j) * (x - j));
							if (dist > r - 1 && dist <= r)
							{
								totalPixel++;
								tempIndex = z * width * height + y * width + x;
								if (d_imagePtr[tempIndex] < darkElementThreshold)
									totalDarkPixel++;
							}
						}
				}

			//face +-x
			for (x = j - r; x <= j + r; x += 2 * r)
				if (x >= 0 && x < width)
				{
					for (z = MAX(s - r, 0); z <= MIN(s + r, slice - 1); z++)
						for (y = MAX(i - r, 0); y <= MIN(i + r, height - 1); y++)
						{
							float dist = sqrtf((z - s) * (z - s) + (y - i) * (y - i) + (x - j) * (x - j));
							if (dist > r - 1 && dist <= r)
							{
								totalPixel++;
								tempIndex = z * width * height + y * width + x;
								if (d_imagePtr[tempIndex] < darkElementThreshold)
									totalDarkPixel++;
							}
						}
				}

			//line +-z/+-y

			for (z = s - r; z <= s + r; z += 2 * r)
				for (y = i - r; y <= i + r; y += 2 * r)
					if (z >= 0 && z < slice && y >= 0 && y < height)
					{
						float dist = sqrtf((z - s) * (z - s) + (y - i) * (y - i) + (x - j) * (x - j));
						if (dist > r - 1 && dist <= r)
						{
							totalPixel--;
							tempIndex = z * width * height + y * width + x;
							if (d_imagePtr[tempIndex] < darkElementThreshold)
								totalDarkPixel--;
						}
					}
			//line +-z/+-x
			for (z = s - r; z <= s + r; z += 2 * r)
				for (x = j - r; x <= j + r; x += 2 * r)
					if (z >= 0 && z < slice && x >= 0 && x < width)
					{
						for (y = MAX(i - r, 0); y <= MIN(i + r, height - 1); y++)
						{
							float dist = sqrtf((z - s) * (z - s) + (y - i) * (y - i) + (x - j) * (x - j));
							if (dist > r - 1 && dist <= r)
							{
								totalPixel--;
								tempIndex = z * width * height + y * width + x;
								if (d_imagePtr[tempIndex] < darkElementThreshold)
									totalDarkPixel--;
							}
						}
					}
			//line +-y/+-x
			for (y = i - r; y <= i + r; y += 2 * r)
				for (x = j - r; x <= j + r; x += 2 * r)
					if (y >= 0 && y < height && x >= 0 && x < width)
					{
						for (z = MAX(s - r, 0); z <= MIN(s + r, slice - 1); z++)
						{
							float dist = sqrtf((z - s) * (z - s) + (y - i) * (y - i) + (x - j) * (x - j));
							if (dist > r - 1 && dist <= r)
							{
								totalPixel--;
								tempIndex = z * width * height + y * width + x;
								if (d_imagePtr[tempIndex] < darkElementThreshold)
									totalDarkPixel--;
							}
						}
					}

			//8 points
			for (z = s - r; z <= s + r; z += 2 * r)
				for (y = i - r; y <= i + r; y += 2 * r)
					for (x = j - r; x <= j + r; x += 2 * r)
					{
						if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= slice)
							continue;
						float dist = sqrtf((z - s) * (z - s) + (y - i) * (y - i) + (x - j) * (x - j));
						if (dist > r - 1 && dist <= r)
						{
							totalPixel++;
							tempIndex = z * width * height + y * width + x;
							if (d_imagePtr[tempIndex] < darkElementThreshold)
								totalDarkPixel++;
						}
					}
		}
	
	//最早写的覆盖

	//if (r == 1)
	//{
	//	for (z = MAX(s - r, 0); z <= MIN(s + r, slice - 1); z++)
	//		for (y = MAX(i - r, 0); y <= MIN(i + r, height - 1); y++)
	//			for (x = MAX(j - r, 0); x <= MIN(j + r, width - 1); x++)
	//			{
	//				tempIndex = z * width * height + y * width + x;
	//				totalPixel++;
	//				if (d_imagePtr[tempIndex] < darkElementThreshold)
	//					totalDarkPixel++;
	//			}
	//}
	//else
	//{
	//	for (z = s - r; z <= s + r; z += 2 * r)
	//		for (y = i - r; y <= i + r; y += 2 * r)
	//			if (z >= 0 && z < slice && y >= 0 && y < height)
	//			{
	//				for (x = MAX(j - r, 0); x <= MIN(j + r, width - 1); x++)
	//				{
	//					tempIndex = z * width * height + y * width + x;
	//					totalPixel++;
	//					if (d_imagePtr[tempIndex] < darkElementThreshold)
	//						totalDarkPixel++;
	//				}
	//			}

	//	for (z = s - r; z <= s + r; z += 2 * r)
	//		for (x = j - r; x <= j + r; x += 2 * r)
	//			if (z >= 0 && z < slice && x >= 0 && x < width)
	//			{
	//				for (y = MAX(i - r, 0); y <= MIN(i + r, height - 1); y++)
	//				{
	//					tempIndex = z * width * height + y * width + x;
	//					totalPixel++;
	//					if (d_imagePtr[tempIndex] < darkElementThreshold)
	//						totalDarkPixel++;
	//				}
	//			}

	//	for (y = i - r; y <= i + r; y += 2 * r)
	//		for (x = j - r; x <= j + r; x += 2 * r)
	//			if (y >= 0 && y < height && x >= 0 && x < width)
	//			{
	//				for (z = MAX(s - r, 0); z <= MIN(s + r, slice - 1); z++)
	//				{
	//					tempIndex = z * width * height + y * width + x;
	//					totalPixel++;
	//					if (d_imagePtr[tempIndex] < darkElementThreshold)
	//						totalDarkPixel++;
	//				}
	//			}
	//}

		if (r >= 1 && totalDarkPixel > coverRatio  * totalPixel) break;
		r++;
	}
	//d_radiusMat[fullIdx] = r;
	d_radiusMat_compact[smallIdx] = r;
	//以后改成smallIdx
}


//calcRadius: 通过d_imagePtr计算节点半径，返回radiusMat和d_radiusMat
void calcRadius_gpu_compact(uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, uchar* d_radiusMat_compact, int width, int height, int slice, int newSize, int globalThreshold)
{
	//d_radiusMat 和 d_imagePtr 来自外部
	cudaError_t errorCheck;

	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "Before calcRadius GPU: " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	calcRadiusKernel_compact << <(newSize-1)/256 + 1, 256 >> > (d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_radiusMat_compact, width, height, slice, newSize, globalThreshold);

	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "In calcRadius GPU: " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}
}

__global__
void FMRadius01(uchar* d_imagePtr, int* d_compress, int* d_decompress,  int* distPtr, int* updatePtr, uchar* d_curStatus, int width, int height, int slice, int newSize)
{
	int smallIdx = threadIdx.x + blockIdx.x*blockDim.x;
	if (smallIdx >= newSize) return;

	if (d_curStatus[smallIdx] != ACTIVE) return;

	//int darkElementThreshold = 5;
	d_curStatus[smallIdx] = PASSIVE;

	int fullIdx = d_decompress[smallIdx];

	int3 curPos;
	curPos.z = fullIdx / (width * height);
	curPos.y = fullIdx % (width * height) / width;
	curPos.x = fullIdx % width;
	//printf("%d %d %d\n", curPos.x, curPos.y, curPos.z);

	int3 neighborPos;
	int neighborfullIdx;
	int neighborSmallIdx;
	for (int k = 0; k < 6; k++)
	{
		neighborPos.x = curPos.x + dx3dconst[k];
		neighborPos.y = curPos.y + dy3dconst[k];
		neighborPos.z = curPos.z + dz3dconst[k];
		if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
			|| neighborPos.z < 0 || neighborPos.z >= slice)
			continue;
		neighborfullIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;
		if (d_imagePtr[neighborfullIdx] == 0) continue;
		neighborSmallIdx = d_compress[neighborfullIdx];
		atomicMin(&updatePtr[neighborSmallIdx], distPtr[smallIdx] + 1);
	}
}

__global__
void FMRadius02(uchar* d_imagePtr, int* distPtr, int* updatePtr, uchar* d_curStatus, int width, int height, int slice, int newSize, int* d_changeFlag)
{
	int smallIdx = threadIdx.x + blockIdx.x*blockDim.x;
	if (smallIdx >= newSize) return;
	if (updatePtr[smallIdx] < distPtr[smallIdx])
	{
		distPtr[smallIdx] = updatePtr[smallIdx];
		d_curStatus[smallIdx] = ACTIVE;
		d_changeFlag[0] = 1;
	}
}


__global__ void convertRadius(int* d_decompress, uchar* d_radiusMat, int* d_distPtr, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;
	int fullIdx = d_decompress[smallIdx];
	int r = d_distPtr[smallIdx];
	if (r <= 1e9 && r <= 255)
	{
		d_radiusMat[fullIdx] = r/1.732;
	}
	else if (r <= 1e9)
		d_radiusMat[fullIdx] = 255/1.732;
	else
		d_radiusMat[fullIdx] = 0;
}

__global__ void calcRadius_Preprocess(uchar* d_imagePtr, int* d_decompress, int* d_distPtr, uchar* d_statusMat, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;
	int fullIdx = d_decompress[smallIdx];
	if (d_imagePtr[fullIdx] == 1)
	{
		d_distPtr[smallIdx] = 1;
		d_statusMat[smallIdx] = ACTIVE;
	}
}



//calcRadius_fastmarching: 通过d_imagePtr+ fastmarching计算节点半径，返回radiusMat和d_radiusMat
void calcRadius_gpu_fastmarching(uchar* d_imagePtr, int* d_compress, int* d_decompress, uchar* d_radiusMat, int width, int height, int slice, int newSize)
{

	double meanValue = 0;
	uchar value = 0;

	int* d_distPtr;
	int* d_upDate;
	uchar * d_curStatus;
	int changeFlag[1] = { 0 };
	int* d_changeFlag;

	cudaMalloc((void**)&d_distPtr, sizeof(int) * newSize);
	cudaMalloc((void**)&d_upDate, sizeof(int) * newSize);
	cudaMalloc((void**)&d_curStatus, sizeof(uchar) * newSize);
	cudaMalloc((void**)&d_changeFlag, sizeof(int));

	thrust::fill(thrust::device, d_distPtr, d_distPtr + newSize, 1e10);
	cudaMemset(d_curStatus, FARAWAY, sizeof(uchar) * newSize);
	
	calcRadius_Preprocess << < (newSize - 1) / 256 + 1, 256 >> > (d_imagePtr, d_decompress, d_distPtr, d_curStatus, newSize);
	cudaMemcpy(d_upDate, d_distPtr, sizeof(int) * newSize, cudaMemcpyDeviceToDevice);


	cudaError_t errorCheck;

	//blockSize:64
	//maxBlockNum:512
	int counter = 0;
	while (1)
	{
		counter++;
		//std::cerr << counter << std::endl;
		FMRadius01 << <(newSize-1)/256+1, 256 >> > (d_imagePtr, d_compress, d_decompress, d_distPtr, d_upDate, d_curStatus,

			width, height, slice, newSize);

		changeFlag[0] = 0;
		cudaMemcpy(d_changeFlag, changeFlag, sizeof(int), cudaMemcpyHostToDevice);

		FMRadius02 << <(newSize - 1) / 256 + 1, 256 >> > (d_imagePtr, d_distPtr, d_upDate, d_curStatus,

			width, height, slice, newSize, d_changeFlag);


		errorCheck = cudaGetLastError();
		if (errorCheck != cudaSuccess) {
			std::cerr << cudaGetErrorString(errorCheck) << std::endl;
			system("pause");
			return;
		}
		cudaMemcpy(changeFlag, d_changeFlag, sizeof(int), cudaMemcpyDeviceToHost);
		if (changeFlag[0] == 0)
			break;

		std::cerr << counter << std::endl;
	}

	convertRadius << < (newSize - 1) / 256 + 1, 256 >> > (d_decompress, d_radiusMat, d_distPtr, newSize);

	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "During Inner FMRadius " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	cudaFree(d_distPtr);
	cudaFree(d_upDate);
	cudaFree(d_curStatus);
	cudaFree(d_changeFlag);
}


__global__ void changeParentKernel_compact(int* d_compress, int* d_decompress, int* d_parentPtr_compact, uchar* d_parentSimplePtr, int width, int height, int slice, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;

	int fullIdx = d_decompress[smallIdx];
	int direction;
	int offset = width * height * slice;
	int z = fullIdx / (width * height);
	int y = (fullIdx % (width * height)) / width;
	int x = fullIdx % width;

	int parentfullIdx, parentSmallIdx;
	int parent2fullIdx, parent2SmallIdx;

	direction = d_parentSimplePtr[smallIdx];
	if (direction != 0xff)
	{
		parentfullIdx  = (z - dz3d26const[direction]) * width * height + (y - dy3d26const[direction]) * width + (x - dx3d26const[direction]);
		parentSmallIdx = d_compress[parentfullIdx];
		d_parentPtr_compact[smallIdx] = parentSmallIdx;
	}
	direction = d_parentSimplePtr[smallIdx + newSize];
	if (direction != 0xff)
	{
		parent2fullIdx = (z - dz3d26const[direction]) * width * height + (y - dy3d26const[direction]) * width + (x - dx3d26const[direction]);
		parent2SmallIdx = d_compress[parent2fullIdx];
		d_parentPtr_compact[smallIdx + newSize] = parent2SmallIdx;
	}
	//for (i = 0; i < seedArr.size(); i++)
	//{
	//	curIdx = seedArr[i];
	//	parentPtr[curIdx] = curIdx;
	//}

}


__global__ void changeSeedParentKernel(int* d_compress, int* d_parentPtr_compact, int* d_seedArr, int seedNum, int width, int height, int slice)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= seedNum) return;
	int curPos = d_seedArr[idx];
	int smallIdx = d_compress[curPos];
	d_parentPtr_compact[smallIdx] = smallIdx;
}



void changeSimpleParentToFull(int* d_compress, int* d_decompress, int* d_parentPtr_compact, uchar* d_parentSimplePtr, std::vector<int>& seedArr, int width, int height, int slice, int newSize)
{
	cudaError_t errorCheck;
	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "Before changeParentKernel " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	changeParentKernel_compact << <(newSize - 1) / 256 + 1, 256 >> > (d_compress, d_decompress, d_parentPtr_compact, d_parentSimplePtr, width, height, slice, newSize);

	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "In changeParentKernel " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	int* d_seedArr;
	int seedNum = seedArr.size();
	cudaMalloc(&d_seedArr, sizeof(int) * seedNum);
	cudaMemcpy(d_seedArr, &(seedArr[0]), sizeof(int) * seedNum, cudaMemcpyHostToDevice);

	changeSeedParentKernel << <(seedNum -1)/256 +1, 256 >> > (d_compress, d_parentPtr_compact, d_seedArr, seedNum, width, height, slice);
	cudaDeviceSynchronize();
	cudaFree(d_seedArr);
	
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "In changeSeedParentKernel " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}
}