#include "mergeSegments.h"
#include "fastmarching.h"
#include "TimerClock.hpp"

////If one want to test the result without merging, use this option.
//#define __NO__MERGE

//disjoint set
//���鼯
__device__ int getfather_gpu(int* d_disjointSet, int x)
{
	if (d_disjointSet[x] == x) return x;
	return d_disjointSet[x] = getfather_gpu(d_disjointSet, d_disjointSet[x]);
}

__device__ void merge_gpu(int* d_disjointSet, uchar* d_seedRadiusMat, int x, int y)
{
	int fa_x = getfather_gpu(d_disjointSet, x);
	int fa_y = getfather_gpu(d_disjointSet, y);
	//�뾶���Ϊ���ף�
	//��ͬ��������С��Ϊ����

	int rx = d_seedRadiusMat[fa_x];
	int ry = d_seedRadiusMat[fa_y];

	if (rx > ry)
	{
		d_disjointSet[fa_y] = fa_x;
	}
	else if (rx < ry)
	{
		d_disjointSet[fa_x] = fa_y;
	}
	else if (rx == ry)
	{
		if (fa_x < fa_y)
			d_disjointSet[fa_y] = d_disjointSet[x];
		else
			d_disjointSet[fa_x] = d_disjointSet[y];
	}
}


/*
����:findInterSectKernel
����:����鿴ĳ�����Ƿ���������֧�Ľ���㣬���ҽ����еĽ�������һ�������С�
�ж��Ƿ��ǽ���㣺����õ��top-2��Ӧ�ķ�֧��top-1��Ӧ�ķ�֧��ͬ(�������fastmarching����)��
���ڶ�����Ҫ����ԭ�Ӳ���,���ѡ����ʹ��share memory����ÿ��block�ڲ�����һ��С�Ͷ��У�����ٺϲ��������С�
mergeSegments()���в���������<50ms�����û�н���������Ϊ��ѹ����
d_seedNumberPtr: ��¼�ýڵ����ĸ�������չ����
*/
/*
Funciton:findInterSectKernel
Work:Checking if a node is an intersect of two neuron branches. All of the intersects are stored into an array(or,queue).
Implementation: If the node's top-2 parent and top-1 parent are extended from different seeds (see fastmarching.cu for detail)
it is regarded as an intersect.
The queue operations are based on atomic operations. We optimized this implementation using atomic operations in shared memory:
in each block, we build a small queue using shared-memory-atomic operations, and combine these small blocks to a global array.
These atomic operations can be replaced by stream compaction, but the whole mergeSegments() function runs very fast, so there 
is little performance gain can be reach by stream compaction.
d_seedNumberPtr: The current voxel is extended from which seed.
*/
__global__ void findInterSectKernel(int * d_compress, int* d_decompress, int* d_parentPtr_compact, short int* d_seedNumberPtr, int width, int height, int slice, int newSize, int* queue, int* queueHead, int* queueLock, int queueMaxSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;
	__shared__ int localQueue[512];
	__shared__ int localQueueHead[1];
	__shared__ int localQueueLock[1];
	__shared__ int offset[1];

	if (threadIdx.x == 0)
	{
		*localQueueHead = 0;
		*localQueueLock = 0;
	}
	__syncthreads();

	int queueloop;
	int parent2SmallIdx = d_parentPtr_compact[smallIdx + newSize];

	if (parent2SmallIdx != -1)
	{
		int ra = d_seedNumberPtr[smallIdx];
		int rb = d_seedNumberPtr[parent2SmallIdx];

		if (ra != 0 && rb != 0 && ra != rb)
		{
			queueloop = 0;
			do {
				if (queueloop = atomicCAS(localQueueLock, 0, 1) == 0)
				{
					int localQsize = localQueueHead[0];
					localQueue[localQsize] = smallIdx;
					localQsize += 1;
					localQueueHead[0] = localQsize;
				}
				__threadfence_block();
				if (queueloop) atomicExch(localQueueLock, 0);
			} while (!queueloop);
		}
	}


	__syncthreads();

	//���������н���ͳ�ƣ���ÿ��С��ֱ����offset
	int localNum = *localQueueHead;
	if (localNum == 0) return;

	if (threadIdx.x == 0 && localNum != 0)
	{
		queueloop = 0;
		do {
			if (queueloop = atomicCAS(queueLock, 0, 1) == 0)
			{
				int qSize = *queueHead;

				if (qSize + localNum < queueMaxSize)
				{
					*offset = qSize;
					qSize += localNum;
					*queueHead = qSize;
				}
				else
				{
					//�Ų����ˣ�������
					*localQueueHead = 0;
				}
			}
			__threadfence();
			if (queueloop) atomicExch(queueLock, 0);
		} while (!queueloop);
	}
	__syncthreads();

	//��share memory����Ķ����������ܶ�����
	//combine the local queues to a global array
	if (threadIdx.x < *localQueueHead)
	{
		queue[*offset + threadIdx.x] = localQueue[threadIdx.x];
	}
}

/*
����:chlcChldKernel
����:�������ÿ�����child����(��������)
*/
//renew the number of child of the nodes
__global__ void calcChildKernel(int* d_compress, int* d_decompress, int* d_parentMat, int* d_childNumPtr, int width, int height, int slice, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;

	int parentSmallIdx = d_parentMat[smallIdx];
	if (parentSmallIdx == -1 || parentSmallIdx == smallIdx) return;
	atomicAdd(d_childNumPtr + parentSmallIdx, 1);
}

/*
����:interSectCheckKernel
����:����鿴ÿ��������Ƿ���Ч��
��Ч�ı�׼�����Top-1 parent�� Top-2 parent ���Ÿ�����Ϣһ·�����Ƿֱ��Ӧ�ĸ�ǰ����
���һ·�����ڵķ�֧��û�з����ı䣬������Ч�ġ������ڽ������ϲ�������֧ʱ������ǣ����������֧�Ӷ���������
*/
/*
Function:interSectCheckKernel
Work:Checking if one intersect is valid.
Implementation��The checking are started from the top-1 and top-2 parent of the intersect. Moving along the chind->parent 
edge until reach the seed point. If all of the voxels along this path belongs to the same seed, the node is a valid intersect.
Otherwise, errors may occur when merging the two branches.
*/
__global__ void interSectCheckKernel(int* d_compress, int* d_decompress, int* d_interSectArr, int interSectNum, uchar* d_interSectValid, int* d_parentMat, short int* d_seedNumberPtr, int* counter, int width, int height, int slice, int newSize)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= interSectNum) return;
	int smallIdx = d_interSectArr[idx];
	
	int parentSmallIdx = d_parentMat[smallIdx];
	int parent2SmallIdx = d_parentMat[smallIdx + newSize];


	int curSeed = d_seedNumberPtr[smallIdx];
	int parentSeed = d_seedNumberPtr[parentSmallIdx];
	int parent2Seed = d_seedNumberPtr[parent2SmallIdx];

	bool checkIfValidInterSect = true;
	int curSmallIdxTemp = smallIdx;
	while(d_parentMat[curSmallIdxTemp] != curSmallIdxTemp)
	{
		int curSeedTemp = d_seedNumberPtr[curSmallIdxTemp];
		if (curSeedTemp != curSeed)
		{
			checkIfValidInterSect = false;
			break;
		}

		curSmallIdxTemp = d_parentMat[curSmallIdxTemp];
	}

	curSmallIdxTemp = parent2SmallIdx;
	while (d_parentMat[curSmallIdxTemp] != curSmallIdxTemp)
	{
		int curSeedTemp = d_seedNumberPtr[curSmallIdxTemp];
		if (curSeedTemp != parent2Seed)
		{
			checkIfValidInterSect = false;
			break;
		}
		curSmallIdxTemp = d_parentMat[curSmallIdxTemp];
	}
	if (checkIfValidInterSect)
	{
		d_interSectValid[idx] = 1;
		atomicAdd(counter, 1);
	}
	else
	{
		d_interSectValid[idx] = 0;
	}

}

/*
����:interSectProcessKernel
����:���ϲ�������֧ʱ����һ����֧��father��Ϣ���¡�
*/
/*
Function:interSectProcessKernel
Work:Renew the "father"(or parent) information of the branches when merging.
*/
__global__ void interSectProcessKernel(int* d_compress, int* d_decompress, int* d_interSectArr, int interSectNum, uchar* d_interSectValid, int* d_parentMat, short int* d_seedNumberPtr, uchar* d_seedRadiusMat, int* d_disjointSet, int width, int height, int slice, int newSize)
{
	if (threadIdx.x != 0) return;
	for (int it = 0; it < interSectNum; it++)
	{
		if (d_interSectValid[it] == 0)
			continue;
		
		int curSmallIdx = d_interSectArr[it];
		int parentSmallIdx = d_parentMat[curSmallIdx];
		int parent2SmallIdx = d_parentMat[curSmallIdx + newSize];

		int curSeed = d_seedNumberPtr[curSmallIdx];
		int parentSeed = d_seedNumberPtr[parentSmallIdx];
		int parent2Seed = d_seedNumberPtr[parent2SmallIdx];

		int father1 = getfather_gpu(d_disjointSet, curSeed);
		int father2 = getfather_gpu(d_disjointSet, parent2Seed);
		int prevIdxTemp, nextIdxTemp, curIdxTemp;
		//printf("%d %d %d %d %d %d\n", it, curIdx, father1, father2, d_radiusMat[father1], d_radiusMat[father2]);
		//father��һ����Ҫ�ϲ�
		//merge branches have different father in the disjoint set
		if (father1 != father2)
		{
			int r1 = d_seedRadiusMat[father1];
			int r2 = d_seedRadiusMat[father2];
			//std::cerr << "Merge:" << father1 << ' ' << father2 << std::endl;
			//�뾶��������С�ĵ���,��parent2��ʼ,�𽥷�����,parnet2��parent��Ϊcur
			//The branch with larger seed radius or smaller seed index becomes the new root of the merged branch.

			//The merging 
			//if (father1 < father2)
			if (r1 > r2 || (r1 == r2 && father1 < father2))
			{

				curIdxTemp = parent2SmallIdx;
				prevIdxTemp = curSmallIdx;
				while (d_parentMat[curIdxTemp] != curIdxTemp) //|| parentSeedMat[curIdxTemp] != father2)
				{
					nextIdxTemp = d_parentMat[curIdxTemp];

					//1.�޸�
					//1.modify
					d_parentMat[curIdxTemp] = prevIdxTemp;
					//2.ǰ��
					//2.forward
					prevIdxTemp = curIdxTemp;
					curIdxTemp = nextIdxTemp;
				}
				//ȷ���ߵ�root��root�����parent�յ����޸�
				//Move along the path until the seed point is reached
				if (d_parentMat[curIdxTemp] == curIdxTemp && curIdxTemp != prevIdxTemp)
				{
					d_parentMat[curIdxTemp] = prevIdxTemp;
				}
			}
			//else if (father1 > father2)
			else
			{
				curIdxTemp = curSmallIdx;
				prevIdxTemp = parent2SmallIdx;

				while (d_parentMat[curIdxTemp] != curIdxTemp) //|| parentSeedMat[curIdxTemp] != father2)
				{
					nextIdxTemp = d_parentMat[curIdxTemp];

					//1.�޸�
					d_parentMat[curIdxTemp] = prevIdxTemp;
					//2.ǰ��
					prevIdxTemp = curIdxTemp;
					curIdxTemp = nextIdxTemp;
				}
				//ȷ���ߵ�root��root�����parent�յ����޸�
				if (d_parentMat[curIdxTemp] == curIdxTemp && curIdxTemp != prevIdxTemp)
				{
					d_parentMat[curIdxTemp] = prevIdxTemp;
				}
			}

			merge_gpu(d_disjointSet, d_seedRadiusMat, father1, father2);
		}
	}
}

//Ϊ�˷�ֹ���鼯�Ľ�������£������е��ٴν���getfather()
//Renew the informatiion in the disjoint set
__global__ void renewColorKernel(int totalColor, int* d_disjointSet)
{
	if (threadIdx.x != 0) return;
	for (int i = 0; i < totalColor; i++)
	{
		getfather_gpu(d_disjointSet, i);
	}
}

__global__
void getSeedRadius(int* d_seedArr, int* d_compress, uchar* d_seedRadiusMat, uchar* d_radiusMat_compact, int totalColor)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= totalColor) return;
	int fullIdx = d_seedArr[idx];
	int smallIdx = d_compress[fullIdx];
	d_seedRadiusMat[idx] = d_radiusMat_compact[smallIdx];
}


__global__
void getExtraIntersect(int* d_compress, int* d_decompress, int* d_childNumMat, int* d_parentPtr_compact, short int* d_seedNumberPtr, int width, int height, int slice, int newSize)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= newSize) return;
	int smallIdx = idx;

	if (d_childNumMat[smallIdx] > 0) return;
	if (d_parentPtr_compact[smallIdx + newSize] != -1) return;
	int fullIdx = d_decompress[smallIdx];



	int3 curPos;
	curPos.z = fullIdx / (width * height);
	curPos.y = fullIdx % (width * height) / width;
	curPos.x = fullIdx % width;

	int3 neighborPos;
	int neighborIdx;
	int neighborSmallIdx;
	uchar neighborValue;

	int curSeed = d_seedNumberPtr[smallIdx];

	//����Χ26��������չ
	//Extend towards 26-way neighbor
	for (int k = 0; k < 6; k++)
	{
		neighborPos.x = curPos.x + dx3dconst[k];
		neighborPos.y = curPos.y + dy3dconst[k];
		neighborPos.z = curPos.z + dz3dconst[k];

		if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
			|| neighborPos.z < 0 || neighborPos.z >= slice)
			continue;

		neighborIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;
		neighborSmallIdx = d_compress[neighborIdx];
		if (neighborSmallIdx == -1)	continue;


		if (d_parentPtr_compact[neighborSmallIdx + newSize] != -1) continue;

		int newSeed = d_seedNumberPtr[neighborSmallIdx];

		if (curSeed != newSeed)
		{
			if (curSeed < newSeed)
			{
				d_parentPtr_compact[smallIdx + newSize] = neighborSmallIdx;
				break;
			}
		}
	}


}

/*
����:mergeSegments
����:�ڳ�ʼ׷�����֮�󣬶��������񾭷�֧���кϲ�
����:���ӵ㼯�� seedArr, ԭͼd_imagePtr, �뾶��Ϣd_radiusMat, 
��֧�������ӵ���Ϣd_seedNumberPtr, ���鼯d_disjointset
*/
/*
Function:mergeSegments
Work:merge the connected branches after constructing initial neuron.
Input: seedArr(The seed set), d_imagePtr(The image), d_radiusMat(The control radius of voxels),
d_seedNumberPtr(The voxel is extended from which seed point), d_disjointset
*/
void mergeSegments(std::vector<int>& seedArr, std::vector<int>& disjointSet, int width, int height, int slice, int newSize, uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, int* d_childNumMat, uchar* d_radiusMat_compact, int* d_parentPtr_compact, short int* d_seedNumberPtr, int* d_disjointSet)
{
	TimerClock timer;
	timer.update();
	std::vector<int> intersectArr;
	cudaError_t errorCheck;

	int* d_seedArr;
	cudaMalloc(&d_seedArr, sizeof(int) * seedArr.size());
	cudaMemcpy(d_seedArr, &(seedArr[0]), sizeof(int) * seedArr.size(), cudaMemcpyHostToDevice);
	int seedNum = seedArr.size();



	//û��:4500
	getExtraIntersect << <(newSize - 1) / 256 + 1, 256 >> > (d_compress, d_decompress, d_childNumMat, d_parentPtr_compact, d_seedNumberPtr, width, height, slice, newSize);
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "After Get Extra Intersect " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	std::cerr << "Get Extra Intersect cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();


	//�ж�һ��ÿ�����㸽���Ƿ���������������

	//01 ����InterSect
	//01 Finding the intersect of branches

	
	const int queueSize = 5000000; //max number of intersects
	int* queue = (int*)malloc(sizeof(int) * queueSize);
	int* d_queue;
	cudaMalloc(&d_queue, sizeof(int) * queueSize);
	int* d_queueHead;
	int* d_queueLock;
	cudaMalloc(&d_queueHead, sizeof(int));
	cudaMalloc(&d_queueLock, sizeof(int));
	cudaMemset(d_queueHead, 0, sizeof(int));
	cudaMemset(d_queueLock, 0, sizeof(int));	
	findInterSectKernel << <(newSize - 1) / 256 + 1, 256 >> > (d_compress, d_decompress, d_parentPtr_compact, d_seedNumberPtr, width, height, slice, newSize, d_queue, d_queueHead, d_queueLock, queueSize);
	int* qSize = (int*)malloc(sizeof(int));
	cudaMemcpy(qSize, d_queueHead, sizeof(int), cudaMemcpyDeviceToHost);
	intersectArr.resize(*qSize);
	cudaMemcpy(&(intersectArr[0]), d_queue, sizeof(int) * (*qSize), cudaMemcpyDeviceToHost);
	int interSectNum = *qSize;

	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "After InterSect Finding " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}
	std::cerr << "InterSect Size: " << interSectNum << std::endl;
	std::cerr << "InterSect Finding cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();




	//02 ���InterSect
	//02 Checking if the intersect is valid (only relate to two branches)
	int countValidInterSect = 0;
	int * d_parentMat_compact = d_parentPtr_compact;

	int* counter = (int*)malloc(sizeof(int));
	int* d_counter;
	cudaMalloc(&d_counter, sizeof(int));
	cudaMemset(d_counter, 0, sizeof(int));
	uchar* d_interSectValid;
	cudaMalloc(&d_interSectValid, sizeof(uchar) * interSectNum);
	cudaMemset(d_interSectValid, 0, sizeof(uchar) * interSectNum);
	interSectCheckKernel << <(interSectNum - 1) / 256 + 1, 256 >> > (d_compress, d_decompress, d_queue, interSectNum, d_interSectValid, d_parentPtr_compact, d_seedNumberPtr, d_counter, width, height, slice, newSize);
	
	
	//cudaMemcpy(counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "After InterSect Checking " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}


	//std::cerr << "Valid InterSect: " << *counter << std::endl;
	std::cerr << "InterSect Chekcing cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();

#ifdef __NO__MERGE
	//�������Բ��ϲ��Ľ����ʹ������Ĵ���
	//If one want to test the result without merging, use this option.
	cudaMemset(d_interSectValid, 0, sizeof(uchar) * interSectNum);
#endif // __NO__MERGE


	//03 Merge
	int totalColor = seedArr.size(); 
	//Hint: The indices of seeds are started from 1. The 0th seed is a dummy seed.
	uchar* d_seedRadiusMat;
	cudaMalloc(&d_seedRadiusMat, sizeof(int) * totalColor);

	getSeedRadius << <(totalColor - 1) / 32 + 1, 32 >> > (d_seedArr, d_compress, d_seedRadiusMat, d_radiusMat_compact, totalColor);


	interSectProcessKernel << <1, 1 >> > (d_compress, d_decompress, d_queue, interSectNum, d_interSectValid, d_parentPtr_compact, d_seedNumberPtr, d_seedRadiusMat, d_disjointSet, width, height, slice, newSize);
	renewColorKernel << <1, 1 >> > (totalColor, d_disjointSet);

	cudaMemcpy(&(disjointSet[0]), d_disjointSet, sizeof(int) * totalColor, cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "After Merge Reverse: " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}
	std::cerr << "Merging cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();

	//04 ����ͳ��childNum
	//04 Renew the number of childs

	cudaMemset(d_childNumMat, 0, sizeof(int) * newSize);
	calcChildKernel << <(newSize - 1) / 256 + 1, 256 >> > (d_compress, d_decompress, d_parentPtr_compact, d_childNumMat, width, height, slice, newSize);
	
	
	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "Renew ChildNum : " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}
	std::cerr << "Renew ChildNum cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();

	cudaFree(d_seedArr);
	cudaFree(d_queue);
	cudaFree(d_queueHead);
	cudaFree(d_queueLock);
	cudaFree(d_counter);
	cudaFree(d_interSectValid);
	cudaFree(d_seedRadiusMat);
	free(queue);
	free(qSize);
	free(counter);

}



//һ�������������ģ��
//a template for checking if two arrays identical
template<typename T>
void crosscheck(const T* d_arr1, const T* d_arr2, int arrSize)
{
	T* res1 = (T*)malloc(sizeof(T) * arrSize);
	T* res2 = (T*)malloc(sizeof(T) * arrSize);
	cudaMemcpy(res1, d_arr1, sizeof(T) * arrSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(res2, d_arr2, sizeof(T) * arrSize, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaError_t errorCheck;
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "In crosscheck: " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}


	int errorCount = 0;
	for (int i = 0; i < arrSize; i++)
	{
		if (res1[i] != res2[i])
		{
			printf("id: %d, res1: %d, res2: %d\n", i, res1[i], res2[i]);
			errorCount++;
		}
		if (errorCount > 500)
			break;
	}
	free(res1);
	free(res2);
	if (errorCount == 0)
		std::cerr << "Crosscheck is clear!" << std::endl;
}
