#include "mergeSegments.h"
#include "fastmarching.h"
#include "TimerClock.hpp"

//#define __NO__MERGE


__device__ int getfather_gpu(int* d_disjointSet, int x)
{
	if (d_disjointSet[x] == x) return x;
	return d_disjointSet[x] = getfather_gpu(d_disjointSet, d_disjointSet[x]);
}

__device__ void merge_gpu(int* d_disjointSet, uchar* d_seedRadiusMat, int x, int y)
{
	int fa_x = getfather_gpu(d_disjointSet, x);
	int fa_y = getfather_gpu(d_disjointSet, y);
	//半径大的为父亲，
	//相同情况下序号小作为父亲

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
函数:findInterSectKernel
功能:逐个查看某个点是否是两个分支的交叉点，并且将所有的交叉点放入一个队列中。
判断是否是交叉点：如果该点的top-2对应的分支和top-1对应的分支不同。
由于队列需要进行原子操作,因此选择了使用share memory，在每个block内部建立一个小型队列，最后再合并成主队列。
mergeSegments()所有操作加起来<50ms，因此没有将互斥锁改为流压缩。
seedNumber: 记录该节点由哪个种子扩展而来
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

	//对整个队列进行统计，对每个小块分别计算offset
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
					//放不下了，不放了
					*localQueueHead = 0;
				}
			}
			__threadfence();
			if (queueloop) atomicExch(queueLock, 0);
		} while (!queueloop);
	}
	__syncthreads();

	//将share memory里面的东西拷贝到总队列中
	if (threadIdx.x < *localQueueHead)
	{
		queue[*offset + threadIdx.x] = localQueue[threadIdx.x];
	}
}

/*
函数:chlcChldKernel
功能:逐个更新每个点的child数量(暴力更新)
*/
__global__ void calcChildKernel(int* d_compress, int* d_decompress, int* d_parentMat, int* d_childNumPtr, int width, int height, int slice, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;

	int parentSmallIdx = d_parentMat[smallIdx];
	if (parentSmallIdx == -1 || parentSmallIdx == smallIdx) return;
	atomicAdd(d_childNumPtr + parentSmallIdx, 1);
}

/*
函数:interSectCheckKernel
功能:逐个查看每个交叉点是否有效。
有效的标准：如果Top-1 parent和 Top-2 parent 沿着父亲信息一路向他们分别对应的根前进，
如果一路上属于的分支都没有发生改变，就是有效的。否则，在接下来合并两个分支时，可能牵扯到其他分支从而发生错误。
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

		//从现在保证valid了
		int father1 = getfather_gpu(d_disjointSet, curSeed);
		int father2 = getfather_gpu(d_disjointSet, parent2Seed);
		int prevIdxTemp, nextIdxTemp, curIdxTemp;
		//printf("%d %d %d %d %d %d\n", it, curIdx, father1, father2, d_radiusMat[father1], d_radiusMat[father2]);
		//father不一样才要合并
		if (father1 != father2)
		{
			int r1 = d_seedRadiusMat[father1];
			int r2 = d_seedRadiusMat[father2];
			//std::cerr << "Merge:" << father1 << ' ' << father2 << std::endl;
			//小的当根,从parent2开始,逐渐反过来,parnet2的parent改为cur
			//if (father1 < father2)
			if (r1 > r2 || (r1 == r2 && father1 < father2))
			{

				curIdxTemp = parent2SmallIdx;
				prevIdxTemp = curSmallIdx;
				while (d_parentMat[curIdxTemp] != curIdxTemp) //|| parentSeedMat[curIdxTemp] != father2)
				{
					nextIdxTemp = d_parentMat[curIdxTemp];

					//1.修改
					d_parentMat[curIdxTemp] = prevIdxTemp;
					//2.前进
					prevIdxTemp = curIdxTemp;
					curIdxTemp = nextIdxTemp;
				}
				//确保走到root后，root本身的parent收到了修改
				if (d_parentMat[curIdxTemp] == curIdxTemp && curIdxTemp != prevIdxTemp)
				{
					d_parentMat[curIdxTemp] = prevIdxTemp;
				}
			}
			//小的当根,从cur开始,逐渐反过来,cur的parent改为parent2
			//else if (father1 > father2)
			else
			{
				curIdxTemp = curSmallIdx;
				prevIdxTemp = parent2SmallIdx;

				while (d_parentMat[curIdxTemp] != curIdxTemp) //|| parentSeedMat[curIdxTemp] != father2)
				{
					nextIdxTemp = d_parentMat[curIdxTemp];

					//1.修改
					d_parentMat[curIdxTemp] = prevIdxTemp;
					//2.前进
					prevIdxTemp = curIdxTemp;
					curIdxTemp = nextIdxTemp;
				}
				//确保走到root后，root本身的parent收到了修改
				if (d_parentMat[curIdxTemp] == curIdxTemp && curIdxTemp != prevIdxTemp)
				{
					d_parentMat[curIdxTemp] = prevIdxTemp;
				}
			}

			merge_gpu(d_disjointSet, d_seedRadiusMat, father1, father2);
		}
	}
}

//为了防止并查集的结果不够新，对所有点再次进行getfather()
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



void mergeSegments(std::vector<int>& seedArr, std::vector<int>& disjointSet, int width, int height, int slice, int newSize, uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, int* d_childNumMat, uchar* d_radiusMat_compact, int* d_parentPtr_compact, short int* d_seedNumberPtr, int* d_disjointSet)
{
	TimerClock timer;
	timer.update();
	std::vector<int> intersectArr;

	int* d_seedArr;
	cudaMalloc(&d_seedArr, sizeof(int) * seedArr.size());
	cudaMemcpy(d_seedArr, &(seedArr[0]), sizeof(int) * seedArr.size(), cudaMemcpyHostToDevice);
	int seedNum = seedArr.size();



	//判断一下每个交点附近是否有他的两个父亲

	//01 查找InterSect

	cudaError_t errorCheck;
	const int queueSize = 5000000; //查找上限,多了就不要了
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
	//std::cerr << "InterSect Size: " << interSectNum << std::endl;
	std::cerr << "InterSect Finding cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();




	//02 检查InterSect
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
	//如果想测试不合并的结果，使用下面的代码
	cudaMemset(d_interSectValid, 0, sizeof(uchar) * interSectNum);
#endif // __NO__MERGE


	//03 Merge
	int totalColor = seedArr.size(); //0号好像不用
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

	//04 重新统计childNum

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



//一个用来检查错误的模板
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
