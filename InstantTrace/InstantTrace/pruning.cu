#include "pruning.h"

//#define __NO__PRUNING
//If one want to try generate resuls without pruning, use this option.

//#define __UNIQUE__COLOR
//If one want to try only keep the central neuron and drop others, use this option.

struct swcPoint
{
	int x, y, z, r;
	int swcIndex, parentSwcIndex;
	int isLeaf, isRoot;
	int seedNumber;

	swcPoint() {};
	swcPoint(int x, int y, int z, int r, int swcIndex, int parentSwcIndex, int seedNumber = 0) :x(x), y(y), z(z), r(r),
		swcIndex(swcIndex), parentSwcIndex(parentSwcIndex), isLeaf(isLeaf), isRoot(isRoot), seedNumber(seedNumber)
	{
	}
};

//01 根据追踪结果(parent数组) ，将整个图划分为不重叠的segment。
//每个segment含有如下信息: leafIdx(叶子)，rootIdx(根), length(总像素数量), parent(父亲分支)，score(评分，用于筛选边)
//01 According to the tracing result (the parent information array), split the whole node-link graph to non-overlapping segments.
//A segment is a path from a leaf to a certain branch point or even seed. There is a one-to-one mapping between leaf nodes and segments.
//A branch point belongs to the longest path who go through this branch point.
//A segment includes these information: leafIdx, rootIdx, length(total pixel number along the path), parent(The higher-level segment), score(a metric used for filter segments)

void constructSegment(std::vector<int>& leafArr, int width, int height, int slice, int newSize, uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, int* d_parentMat_compact, uchar*  d_statusMat_compact, int* d_childNumMat_compact,
	int*& d_segment_leafIdx, int*& d_segment_rootIdx, int*& d_segment_length, int*& d_segment_parent, float*& d_segment_score, int& segNumber, int darkLeafThreshold);

//02 按照score进行剪枝，本步骤会删除绝大多数分支，剩下一些较长的分支
//02 Filter the segments by a threshold of score. This process will filter the most of the branches, and keep the long branches.
void filterSegment(int* d_segment_leafIdx, int* d_segment_rootIdx, int* d_segment_length, int* d_segment_parent, float* d_segment_score, short int* d_seedNumberPtr, int* d_disjointSet, int* d_compress_outer, int totalColor, int scoreThreshold, int segNumber, int& segNumberFiltered);

//04 输出值到SWC文件，需要得到每个点的坐标(x,y,z,半径r, 父亲parent, 颜色color)
//04 Output in *.swc format. Every point have its x,y,z coordinate, control radius r, and color.
void outputSwc(int* d_compress, int* d_decompress, int* d_parentMat_compact, uchar* d_radiusMat_compact, short int* d_seedNumberPtr, int* d_disjointSet, std::vector<int>& segment_leafIdx_final, std::vector<int>& segment_rootIdx_final, std::vector<int>& segment_length_final, int width, int height, int slice, int* d_segment_leafIdx, int* d_segment_rootIdx, int segNumberFiltered, int segNumberFinal, std::string inputName);


/*
函数:travelSegmentKernel
作用:遍历分支，该分支上各个节点的index放入数组，以便下面进行并行处理
*/
/*
Function:travelSegmentKernel
Work: Travel through the whole neuron tree, and store the indices of nodes of segments.
*/

__global__ void travelSegmentKernel(int* d_compress, int* d_decompress, int* d_parentMat_compact, int* d_segment_leafIdx, int* d_segment_rootIdx, int* d_segment_length, int* d_lengthPrefixSum, int* d_pointIdxMat, int segNumberFiltered)
{
	int segId = blockDim.x * blockIdx.x + threadIdx.x;
	if (segId >= segNumberFiltered) return;

	//刚才用scan计算了每个分支的对应起点
	int offset = d_lengthPrefixSum[segId];
	int leafIdx = d_segment_leafIdx[segId];
	int rootIdx = d_segment_rootIdx[segId];
	int fullIdx = leafIdx;
	int smallIdx = d_compress[fullIdx];

	int pointCounter = offset;

	while (1)
	{
		d_pointIdxMat[pointCounter++] = fullIdx;
		if (fullIdx == rootIdx) break;
		smallIdx = d_parentMat_compact[smallIdx];
		fullIdx = d_decompress[smallIdx];
	}
}

/*
函数:fastCheckKernel
作用:根据该分支的parent分支的保留情况，快速判断该分支是否保留
*/
//A fast checking kernel for judging if current segment is to be pruned
__global__ void fastCheckKernel(int* d_isSegKeep, int* d_isParentKeep)
{
	//如果parent保留，那么本分支也暂时保留(iskeep == 1)，交给下面的kernel判断是否丢弃；
	//如果parent已经被丢弃，那么本分支直接标记为丢弃(iskeep == 0)
	//If the parent segment have been pruned (iskeep==0) , the current segment will be pruned too.
	*d_isSegKeep = *d_isParentKeep;

	if (threadIdx.x == 0)
	{
		if (*d_isParentKeep == -1)
			printf("parent not finish\n");
	}
}

/*
函数:calcSegKernel
作用:统计某个分支上各个点是否已经被其他已经保留的分支所覆盖
*/
//Calculate whether the current segment have been covered by other "valid" branches.
__global__
void calcSegKernel(uchar* d_imagePtr, uchar* d_coverImagePtr, int * d_pointIdxMat, int* d_isSegKeep, int start, int length, float* d_sumSigPtr, float* d_sumRdcPtr)
{
	//如果刚才fastCheck的时候已经丢弃了本分支，则无须运行
	if (d_isSegKeep[0] == 0) return;
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx >= length) return;
	if (idx < length)
	{
		int pointIdx = d_pointIdxMat[start + idx];
		float oldValue = d_imagePtr[pointIdx];
		float newValue = d_coverImagePtr[pointIdx];
		//已被覆盖了
		if (oldValue != newValue)
		{
			//Rdc: The redundant signal
			atomicAdd(d_sumRdcPtr, oldValue);
		}
		else
		{
			//Sig: The valid signal
			atomicAdd(d_sumSigPtr, oldValue);
		}
	}
}


/*
函数:changeStatusKernel
作用:根据统计该分支覆盖率的情况，决定某个分支是否保留
*/
//Decide whether current segment is to be pruned according to result of calcSegKernel().
__global__ void changeStatusKernel(int* d_isSegKeep, float* d_sumSigPtr, float* d_sumRdcPtr)
{
	//如果此时为0,说明刚才fastcheck发现该分支的parent被丢弃了，本分支也直接丢弃
	if (d_isSegKeep[0] == 0)
	{
		//printf("fastcheck work!\n");
		d_sumSigPtr[0] = 0;
		d_sumRdcPtr[0] = 0;
		return;
	}
	
	//否则，根据刚才统计的被覆盖区域和未被覆盖区域的比值判断是否保留
	//printf("\nBefore Change Status: %.2f %.2f %d\n", *d_sumSigPtr, *d_sumRdcPtr, d_isSegKeep[0]);
	//When the cover ratio exceed certain threshold, prune it
	if (d_sumRdcPtr[0] < 1 || d_sumSigPtr[0] / d_sumRdcPtr[0] > 1.0f / 9)
	{
		d_isSegKeep[0] = 1; //Keep
	}
	else
	{
		d_isSegKeep[0] = 0;
	}

	//重置
	d_sumSigPtr[0] = 0;
	d_sumRdcPtr[0] = 0;
	//printf("\n After Change Status: %.2f %.2f %d\n", *d_sumSigPtr, *d_sumRdcPtr, d_isSegKeep[0]);
}


/*
函数：deleteSegKernel（新版）
功能：将某个分支覆盖的所有区域在整个图像中进行删除。本kernel为child kernel（使用了动态并行）。
*/
/*
Function：deleteSegKernel
Work：Delete the cover area of each nodes in a segment。This kernel is child kernel, and CUDA dynamic parallism is used.
*/
__global__
void deleteSegKernel_Simple_child(uchar* d_coverImagePtr, int width, int height, int slice, int x0, int y0, int z0, int r)
{
	int xPos = blockDim.x * blockIdx.x + threadIdx.x + x0 - r;
	int yPos = blockDim.y * blockIdx.y + threadIdx.y + y0 - r;
	int zPos = blockDim.z * blockIdx.z + threadIdx.z + z0 - r;

	if (xPos < 0 || yPos <0 || zPos < 0 || xPos >= width || yPos >= height || zPos >= slice) return;

	if ((xPos - x0) * (xPos - x0) + (yPos - y0) * (yPos - y0) + (zPos - z0) * (zPos - z0) <= r * r)
		d_coverImagePtr[zPos * width * height + yPos * width + xPos] = 0;
}


__global__
void deleteSegKernel_Simple(int* d_compress, uchar* d_coverImagePtr, uchar* d_radiusMat_compact, int * d_pointIdxMat, int* d_isSegKeep, int start, int length, int width, int height, int slice, int* d_groupStartPos, int* d_groupStartOffset, int* d_groupEndPos, int* d_groupEndOffset)
{
	if (d_isSegKeep[0] == 0) return;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= length) return;

	int pointIdx = d_pointIdxMat[start + idx];
	int smallIdx = d_compress[pointIdx];
	int r = d_radiusMat_compact[smallIdx];

	int z0 = pointIdx / (width * height);
	int y0 = pointIdx % (width * height) / width;
	int x0 = pointIdx % width;

	dim3 grid(2 * r / 16 + 1, 2 * r /16 + 1, 2 * r + 1);
	dim3 block(16, 16, 1);

	deleteSegKernel_Simple_child << < grid, block >> > (d_coverImagePtr, width, height, slice, x0, y0, z0, r);
}



/*
函数：darkSegmentFilterKernel
功能：判断某个分支是否过暗，如果过暗，直接舍去
*/
/*
Function：darkSegmentFilterKernel
Work：If a segment is dark enough, prune it
*/
__global__ void darkSegmentFilterKernel(uchar* d_imagePtr, int* d_parentMat_compact, int* d_compress, int* d_decompress, int _leafIdx, int _rootIdx, int* d_isSegKeep, int darkSegmentThreshold, int darkLeafThreshold)
{
	//如果刚才的fastCheck中直接丢弃了,那么无须运行
	if (d_isSegKeep[0] == 0) return;

	int leafIdx = _leafIdx;
	int rootIdx = _rootIdx;
	int currentIdx = leafIdx;
	int smallIdx = d_compress[currentIdx];

	int nodeCount = 0;
	int darkNodeCount = 0;
	double sumValue = 0;
	while (1)
	{
		double value = d_imagePtr[currentIdx];
		nodeCount++; sumValue += value;
		if (value < darkSegmentThreshold)
			darkNodeCount++;
		if (currentIdx == rootIdx) break;

		smallIdx = d_parentMat_compact[smallIdx];
		currentIdx = d_decompress[smallIdx];
	}

	double darkSegmentRatio = 0.8;
	if (sumValue / (nodeCount * darkLeafThreshold) < 1 || darkNodeCount > darkSegmentRatio * nodeCount)
	{
		d_isSegKeep[0] = 0;
		//printf("DarkSegment Work!, leaf:%d, root:%d\n", leafIdx, rootIdx);
	}
}



//The main function for pruning the reconstruction result after initial neuron tracing and merging.
void pruneLeaf_3d_gpu(std::vector<int>& leafArr, int &validLeafCount, std::vector<int> & disjointSet, int width, int height, int slice, int newSize, uchar* d_radiusMat_compact, uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, int* d_parentMat_compact, uchar*  d_statusMat_compact, int* d_childNumMat_compact, short int* d_seedNumberPtr, int * d_disjointSet, std::string inputName)
{

	cudaError_t errorCheck;

	TimerClock timer, timer2;
	timer.update();
	timer2.update();

	int darkLeafThreshold = 2;

	int* d_segment_leafIdx;
	int* d_segment_rootIdx;
	int* d_segment_length;
	int* d_segment_parent;
	float* d_segment_score;
	int segNumber;

	//01 根据追踪结果(parent数组) ，将整个图划分为不重叠的segment。
	//01 According to the tracing result (the parent information array), split the whole node-link graph to non-overlapping segments.
	//A segment is a path from a leaf to a certain branch point or even seed. There is a one-to-one mapping between leaf nodes and segments.
	//A branch point belongs to the longest path who go through this branch point.
	//A segment includes these information: leafIdx, rootIdx, length(total pixel number along the path), parent(The higher-level segment), score(a metric used for filter segments)


	constructSegment(leafArr, width, height, slice, newSize, d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_parentMat_compact, d_statusMat_compact, d_childNumMat_compact,
		d_segment_leafIdx, d_segment_rootIdx, d_segment_length, d_segment_parent, d_segment_score, segNumber, darkLeafThreshold);

	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "During construct segment Error:" << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	std::cerr << "First construct segment: " << segNumber << std::endl;
	std::cerr << "Construct Segment cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();



	//02 按照score进行剪枝，本步骤会删除绝大多数分支，剩下一些较长的分支
	//02 Filter the segments by a threshold of score. This process will filter the most of the branches, and keep the long branches.

	int lengthThreshold = 5;

	lengthThreshold = 50;
	//lengthThreshold = 10; //for flycircuits

	

#ifdef __NO__PRUNING
	lengthThreshold = 1;
	//lengthThreshold = 50;
#else

#endif // __NO__PRUING



	int scoreThreshold = lengthThreshold;
	int segNumberFiltered;

	int totalColor = disjointSet.size() - 1;

	filterSegment(d_segment_leafIdx, d_segment_rootIdx, d_segment_length, d_segment_parent, d_segment_score, d_seedNumberPtr, d_disjointSet, d_compress, totalColor, scoreThreshold, segNumber, segNumberFiltered);

	std::vector<int> segment_leafIdx_filtered(segNumberFiltered, -1);
	std::vector<int> segment_rootIdx_filtered(segNumberFiltered, -1);
	std::vector<int> segment_length_filtered(segNumberFiltered, -1);
	std::vector<int> segment_parent_filtered(segNumberFiltered, -1);

	cudaMemcpy(&segment_leafIdx_filtered[0], d_segment_leafIdx, sizeof(int) * segNumberFiltered, cudaMemcpyDeviceToHost);
	cudaMemcpy(&segment_rootIdx_filtered[0], d_segment_rootIdx, sizeof(int) * segNumberFiltered, cudaMemcpyDeviceToHost);
	cudaMemcpy(&segment_length_filtered[0], d_segment_length, sizeof(int) * segNumberFiltered, cudaMemcpyDeviceToHost);
	cudaMemcpy(&segment_parent_filtered[0], d_segment_parent, sizeof(int) * segNumberFiltered, cudaMemcpyDeviceToHost);

	std::cerr << "pruned: " << segNumber << " - " << segNumberFiltered << " = " << segNumber - segNumberFiltered << std::endl;
	std::cerr << "Score filter cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();
	
	std::cerr << "After length pruning: " << segNumberFiltered << std::endl;

	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "After length pruning Error:" << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}


	//03 按照覆盖区域进行剪枝。当选择保留一个分支时，将它的所有影响区域在整个图中减去；
	//判断是否保留分支，首先看它的父亲分支是否被保留(如果父亲丢弃了，儿子也要丢弃）,
	//否则判断它被覆盖的程度，如果覆盖过多则丢弃，否则保留。
	//03 Pruning according to the covering area. When choose to keep a segment, delete all of its cover area in the image.
	//Whether a segment is pruned or kept is according to both its parent segment's status and its covering status.


	double total_length = 0;
	int temp_length = 0;

	int* segStartPosMat = (int*)malloc(sizeof(int) * segNumberFiltered);
	int* segLengthMat = (int*)malloc(sizeof(int) * segNumberFiltered);	
	int pointCounter = 0;


	total_length = 0;
	for (int i = 0; i < segNumberFiltered; i++)
	{
		int leafIdx = segment_leafIdx_filtered[i];
		int rootIdx = segment_rootIdx_filtered[i];
		temp_length = segment_length_filtered[i];
		segStartPosMat[i] = total_length;
		segLengthMat[i] = temp_length;
		total_length += temp_length;
	}


	int* d_pointIdxMat;
	cudaMalloc(&d_pointIdxMat, sizeof(int) * ((int)total_length + 1));

	int* d_lengthPrefixSum;
	cudaMalloc(&d_lengthPrefixSum, sizeof(int) * segNumberFiltered);
	cudaMemset(d_lengthPrefixSum, 0, sizeof(int) * segNumberFiltered);

	thrust::exclusive_scan(thrust::device, d_segment_length, d_segment_length + segNumberFiltered, d_lengthPrefixSum);
	travelSegmentKernel << <(segNumberFiltered -1)/16 + 1, 16 >> > (d_compress, d_decompress, d_parentMat_compact, d_segment_leafIdx, d_segment_rootIdx, d_segment_length, d_lengthPrefixSum, d_pointIdxMat, segNumberFiltered);

	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "After scan Error:" << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}
	std::cerr << "Travel Segment Cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();



	std::vector<int> isSegKeep(segNumberFiltered, 0);
	int* d_isSegKeep; 
	uchar* d_coverImagePtr; 
	cudaMalloc((void **)&d_isSegKeep, sizeof(int) * segNumberFiltered);
	cudaMalloc((void **)&d_coverImagePtr, sizeof(uchar) * width * height * slice);
	float* d_sumSigPtr;
	float* d_sumRdcPtr;
	cudaMalloc((void **)&d_sumSigPtr, sizeof(float) * segNumberFiltered);
	cudaMalloc((void **)&d_sumRdcPtr, sizeof(float) * segNumberFiltered);

	thrust::fill(thrust::device, d_sumSigPtr, d_sumSigPtr + segNumberFiltered, 0.0f);
	thrust::fill(thrust::device, d_sumRdcPtr, d_sumRdcPtr + segNumberFiltered, 0.0f);

	cudaMemcpy(d_coverImagePtr, d_imagePtr, sizeof(uchar) * width * height * slice, cudaMemcpyDeviceToDevice);
	cudaMemset(d_isSegKeep, 0xff, sizeof(int) * segNumberFiltered);

	

	//
	std::vector<int> indegreeMat(segNumberFiltered, 0);
	std::vector<std::vector<int>> outEdgeMat(segNumberFiltered, std::vector<int>(1,0));
	//new segment_parent:  parent of lengthFiltered Segment Idx 

	for (int i = 0; i < segNumberFiltered; i++)
	{
		int parent= segment_parent_filtered[i];
		if (parent != -1)
		{
			//std::cerr << "Parent" << parent << std::endl;
			outEdgeMat[parent].push_back(i);
			outEdgeMat[parent][0] ++;
			indegreeMat[i]++;
			//if (segment_length_filtered[parent] <= segment_length_filtered[i])
			//	std::cerr << "Parent is no longer than child! " << i << ' ' << parent << std::endl;
		}
	}

	std::cerr << "After Calc Topsort" << std::endl;


	int* d_groupStartPos;
	int* d_groupStartOffset;
	int* d_groupEndPos;
	int* d_groupEndOffset;
	cudaMalloc(&d_groupStartPos, sizeof(int) * segNumberFiltered);
	cudaMalloc(&d_groupStartOffset, sizeof(int) * segNumberFiltered);
	cudaMalloc(&d_groupEndPos, sizeof(int) * segNumberFiltered);
	cudaMalloc(&d_groupEndOffset, sizeof(int) * segNumberFiltered);



	int finishflag = 0;
	int turn = 0;
	int solved = 0;

	int filterByDarkSegment = 0;
	int darkSegmentThreshold = 15;//30
	darkSegmentThreshold = 5;
	
	int start;
	int length;
	std::vector<int> indegreeMat_LastTurn;
	while (!finishflag)
	{
		//std::cerr << "Turn: " << turn << std::endl << std::endl << std::endl;
		finishflag = 1;
		turn++;

		//vector operator= is deepcopy
		indegreeMat_LastTurn = indegreeMat;

		for (int currentSeg = 0; currentSeg < segNumberFiltered; currentSeg++)
		{

			//cudaMemset(d_sumSigPtr, 0, sizeof(float));
			//cudaMemset(d_sumRdcPtr, 0, sizeof(float));

			if (indegreeMat_LastTurn[currentSeg] == 0)
			{
				start = segStartPosMat[currentSeg];
				length = segLengthMat[currentSeg];
				//std::cerr << "current: " << currentSeg << std::endl;
				//std::cerr << "start: " << start << " length: " << length << std::endl;
				dim3 block(256, 1, 1);
				dim3 grid((length + 1 - 1) / 256 + 1, 1, 1);

				finishflag = 0;
				int parentSeg = segment_parent_filtered[currentSeg];
				//std::cerr << "CurrentSeg: " << currentSeg << " ParentSeg: " << parentSeg << std::endl;
				//std::cerr << "CurrentSeg:" << currentSeg << " ParentSeg: " << parentSeg << std::endl;
				if (parentSeg != -1)
				{
					fastCheckKernel << <1, 1 >> > (d_isSegKeep + currentSeg, d_isSegKeep + parentSeg);//mapped memory is faster
				}	

				int _leafIdx = segment_leafIdx_filtered[currentSeg];
				int _rootIdx = segment_rootIdx_filtered[currentSeg];

				darkSegmentFilterKernel << <1, 1 >> > (d_imagePtr, d_parentMat_compact, d_compress, d_decompress, _leafIdx, _rootIdx, d_isSegKeep + currentSeg, darkSegmentThreshold, darkLeafThreshold);

				calcSegKernel << <grid, block >> > (d_imagePtr, d_coverImagePtr, d_pointIdxMat, d_isSegKeep + currentSeg, start, length, d_sumSigPtr + currentSeg, d_sumRdcPtr + currentSeg);

				changeStatusKernel << <1, 1 >> > (d_isSegKeep + currentSeg, d_sumSigPtr + currentSeg, d_sumRdcPtr + currentSeg);

				deleteSegKernel_Simple << <(length -1) / 32 + 1, 32 >> > (d_compress, d_coverImagePtr, d_radiusMat_compact, d_pointIdxMat, d_isSegKeep + currentSeg, start, length, width, height, slice, d_groupStartPos, d_groupStartOffset, d_groupEndPos, d_groupEndOffset);



				cudaDeviceSynchronize();
				errorCheck = cudaGetLastError();
				if (errorCheck != cudaSuccess) {
					std::cerr << "In delete kernel synchronize " << cudaGetErrorString(errorCheck) << std::endl;
					system("pause");
					return;
				}

				for (int i = 0; i < outEdgeMat[currentSeg][0]; i++)
				{
					indegreeMat[outEdgeMat[currentSeg][i + 1]] -= 1;
				}
				//标记为-1表示已经结束了
				indegreeMat[currentSeg] = -1;
				solved++;
			}

		}
		std::cerr << "solved:" << solved << std::endl;
	}
	std::cerr << "total solved:" << solved <<  std::endl;


	cudaMemcpy(&(isSegKeep[0]), d_isSegKeep, sizeof(int) * segNumberFiltered, cudaMemcpyDeviceToHost);

	free(segLengthMat);
	cudaFree(d_groupStartPos);
	cudaFree(d_groupStartOffset);
	cudaFree(d_groupEndPos);
	cudaFree(d_groupEndOffset);
	free(segStartPosMat);
	cudaFree(d_isSegKeep);
	cudaFree(d_pointIdxMat);
	cudaFree(d_coverImagePtr);
	cudaFree(d_sumSigPtr);
	cudaFree(d_sumRdcPtr);

	std::cerr << "Pruning GPU Part Cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();

#ifdef __NO__PRUNING
	for (int i = 0; i < segNumberFiltered; i++)
		isSegKeep[i] = 1;

#else

#endif


	int segNumberFinal = 0;
	for (int i = 0; i < segNumberFiltered; i++)
	{
		int leafIdx = segment_leafIdx_filtered[i];//seg->leafIdx;
		int rootIdx = segment_rootIdx_filtered[i];//seg->rootIdx;
		int parentSegIdx = segment_parent_filtered[i];

		if (isSegKeep[i] != 0 && isSegKeep[i] != 1)
		{
			std::cerr << "A seg is not solved!" << std::endl;
			std::cerr << "curSeg: " << i<< " parentSeg: " << parentSegIdx << std::endl;
		}

		if (parentSegIdx == -1)
		{

		}
		else
		{
			if (isSegKeep[i] == 1 && isSegKeep[parentSegIdx] == 0)
			{
				std::cerr << "Violation of Topsort!" << std::endl;
				std::cerr << "curSeg: " << i << " parentSeg: " << parentSegIdx << std::endl;
			}
		}

		if (isSegKeep[i] == 0)
		{
		
		}
		else
		{
			segNumberFinal++;
		}
		

	}


	std::vector<int> segment_leafIdx_final(segNumberFinal, -1);
	std::vector<int> segment_rootIdx_final(segNumberFinal, -1);
	std::vector<int> segment_length_final(segNumberFinal, -1);
	std::vector<int> segment_parent_final(segNumberFinal, -1);

	int counter = 0;
	for (int i = 0; i < segNumberFiltered; i++)
	{
		if (!isSegKeep[i])
		{
			continue;
		}

		segment_leafIdx_final[counter] = segment_leafIdx_filtered[i];
		segment_rootIdx_final[counter] = segment_rootIdx_filtered[i];
		segment_length_final[counter] = segment_length_filtered[i];

		//parent是特殊的，应该用new_seg_parent
		//segment_parent_final[counter] = segment_parent_filtered[i];
		counter++;
	}

	std::cerr << "Pruning CPU Part Cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();


	std::cerr << "prune by coverage (segment number) : " << segNumberFiltered << " - " << segNumberFinal << " = " << segNumberFiltered - segNumberFinal << std::endl;
	validLeafCount = segNumberFinal;
	std::cerr << "Pruning total Cost: " << timer2.getTimerMilliSec() << "ms" << std::endl;
	timer2.update();


	

	//04 输出值到SWC文件，需要得到每个点的坐标(x,y,z,半径r, 父亲parent, 颜色color)
	//output to *.swc format

	

	outputSwc(d_compress, d_decompress, d_parentMat_compact, d_radiusMat_compact, d_seedNumberPtr, d_disjointSet, segment_leafIdx_final, segment_rootIdx_final, segment_length_final, width, height, slice, d_segment_leafIdx, d_segment_rootIdx, segNumberFiltered, segNumberFinal, inputName);
	std::cerr << "Output SWC Cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();



	cudaFree(d_segment_leafIdx);
	cudaFree(d_segment_parent);
	cudaFree(d_segment_rootIdx);
	cudaFree(d_segment_score);
	cudaFree(d_segment_length);

}



/*
函数:findLeafLocalQueueKernel
功能:逐个查看某个点是否是叶子节点(没有child)，并且将所有的叶子放入一个队列中。
由于队列需要进行原子操作,因此选择了使用share memory，在每个block内部建立一个小型队列，最后再合并成主队列。
*/
/*
Function:findLeafLocalQueueKernel
Work:find all of the leaves in the tracing result, and put them into a queue. Shared memory and local queues are used for optimizaion.
*/


__global__ void findLeafLocalQueueKernel(uchar * d_imagePtr, int* d_decompress, uchar* d_statusMat_compact, int* d_childNumMat_compact, int width, int height, int slice, int newSize, int* queue, int* queueHead, int* queueLock, int queueMaxSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;
	__shared__ int localQueue[512];
	__shared__ int localQueueHead[1];
	__shared__ int localQueueLock[1];
	__shared__ int startPos[1];

	if (threadIdx.x == 0)
	{
		*localQueueHead = 0;
		*localQueueLock = 0;
	}
	__syncthreads();

	int queueloop;

	//ALIVE代表之前追踪的时候访问过了
	if (d_statusMat_compact[smallIdx] == ALIVE && d_childNumMat_compact[smallIdx] == 0)
	{
		queueloop = 0;
		do {
			if (queueloop = atomicCAS(localQueueLock, 0, 1) == 0)
			{
				int qhead = localQueueHead[0];
				localQueue[qhead] = smallIdx;
				qhead += 1;
				localQueueHead[0] = qhead;
			}
			__threadfence_block();
			if (queueloop) atomicExch(localQueueLock, 0);
		} while (!queueloop);
	}


	__syncthreads();


	int localNum = *localQueueHead;
	if (localNum == 0) return;

	if (threadIdx.x == 0 && localNum != 0)
	{
		queueloop = 0;
		do {
			if (queueloop = atomicCAS(queueLock, 0, 1) == 0)
			{
				int qhead = *queueHead;

				if (qhead + localNum < queueMaxSize)
				{
					*startPos = qhead;
					qhead += localNum;
					*queueHead = qhead;
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

	if (threadIdx.x < *localQueueHead)
	{
		queue[*startPos + threadIdx.x] = localQueue[threadIdx.x];
	}

}


/*
函数:findDarkLeafKernel
功能: 将过暗的叶子节点删除，并重复这个动作，直到遇到分叉点或者足够亮的节点。
本kernel的作用是从每个叶子开始向上查找，将这些需要删除的节点标记。
*/
//Delete the leaves that is dark, and iterate this process until a branch point or bright point is met.
__global__ void findDarkLeafKernel(uchar * d_imagePtr_compact, int* d_compress, int* d_decompress, uchar* d_statusMat_compact, int* d_childNumMat_compact, int* d_parentMat_compact, int* queue, int* queueHead, int darkLeafThreshold)
{
	int qhead = *queueHead;
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	for (int start = tid; start < qhead; start += blockDim.x * gridDim.x)
	{
		int smallIdx = queue[start];
		//这里写<=1。之前版本是一直找child为0的；实际上，走到分叉点之前的所有过暗点都可以删除，因此这里写<=1

		while (d_imagePtr_compact[smallIdx] < darkLeafThreshold && d_statusMat_compact[smallIdx] == ALIVE && d_childNumMat_compact[smallIdx] <= 1)
		{
			d_statusMat_compact[smallIdx] = DARKLEAF_PRUNED; //作为delete_flag
			if (d_parentMat_compact[smallIdx] == smallIdx || d_parentMat_compact[smallIdx] == -1) break;
			int parentSmallIdx = d_parentMat_compact[smallIdx];

			//d_childNumMat[parentIdx] -= 1; //需要在别的地方处理
			//d_parentMat[curIdx] = -1;/需要在别的地方处理
			smallIdx = parentSmallIdx;
		}
	}
}

/*
函数:pruneDarkLeafKernel
功能: 将过暗的叶子节点删除，并重复这个动作，直到遇到分叉点或者足够亮的节点。
本kernel的作用是删除上一个kernel标记了的节点。
*/
//Delete the points marked by the prevous kernel.
__global__ void pruneDarkLeafKernel(uchar * d_imagePtr_compact, int* d_compress, int* d_decompress, int newSize, uchar* d_statusMat_compact, volatile int* d_childNumMat_compact, int* d_parentMat_compact, int darkLeafThreshold, int* lockArr, int lockArrSize)
{
	int smallIdx = threadIdx.x + blockDim.x * blockIdx.x;
	if (smallIdx >= newSize) return;

	int parentSmallIdx;
	int queueloop;

	if (d_statusMat_compact[smallIdx] == DARKLEAF_PRUNED)
	{
		parentSmallIdx = d_parentMat_compact[smallIdx];
		if (parentSmallIdx == -1)
			printf("the parent of %d %d is -1\n", smallIdx);

		if (d_imagePtr_compact[parentSmallIdx] >= darkLeafThreshold)
		{

			int parentMap = parentSmallIdx % lockArrSize;

			queueloop = 0;

			do {
				if (queueloop = atomicCAS(lockArr + parentMap, 0, 1) == 0)
				{
					d_childNumMat_compact[parentSmallIdx] -= 1;
				}
				__threadfence();
				if (queueloop) atomicExch(lockArr + parentMap, 0);
			} while (!queueloop);

		}
		d_childNumMat_compact[smallIdx] = 0;
		d_parentMat_compact[smallIdx] = -1;
		d_statusMat_compact[smallIdx] = FAR;
	}
}



__global__ void childNumRenewKernel(uchar * d_imagePtr, uchar* d_tempChildNumMat, int* d_parentMat, uchar* d_visited, int* queue, int* queueHead)
{
	int qhead = *queueHead;
	for (int start = 0; start < qhead; start++)
	{
		int index = queue[start];
		int parentIndex;
		if (d_visited[index])
		{
			printf("Logical Error!\n");
		}

		d_visited[index] = 1;
		while (d_parentMat[index] != index)
		{
			parentIndex = d_parentMat[index];
			if (parentIndex == -1)
				printf("ParentIndex equals -1\n");
			d_tempChildNumMat[parentIndex] += 1;
			index = parentIndex;
			if (d_visited[index])
				break;
			d_visited[index] = 1;
		}
	}
}


__global__
static void calcChildKernel(int* d_compress, int* d_decompress, int* d_parentMat, int* d_childNumPtr, uchar* d_statusMat_compact, int width, int height, int slice, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;

	if (d_statusMat_compact[smallIdx] == ALIVE)
	{
		int parentSmallIdx = d_parentMat[smallIdx];
		if (parentSmallIdx == -1 || parentSmallIdx == smallIdx) return;
		atomicAdd(d_childNumPtr + parentSmallIdx, 1);
	}
}


__global__ void childNumRenewKernel_faster(uchar * d_imagePtr, int* d_compress, int* d_decompress, int* d_childNumMat_compact, int* d_parentMat_compact, uchar* d_visited, int* queue, int seedNum, int* lockArr, int lockArrSize)
{

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= seedNum) return;

	int smallIdx = queue[idx];
	int parentSmallIdx;
	if (d_visited[smallIdx])
	{
		printf("Logical Error!\n");
	}

	d_visited[smallIdx] = 1;
	__threadfence();

	int queueloop;
	bool exitflag = false;


	while (d_parentMat_compact[smallIdx] != smallIdx && (!exitflag))
	{
		parentSmallIdx = d_parentMat_compact[smallIdx];

		int parentMap = parentSmallIdx % lockArrSize;

		queueloop = 0;

		do {
			if (queueloop = atomicCAS(lockArr + parentMap, 0, 1) == 0)
			{
				//以下的三个数组都是可变的，因此放在Atomic里面
				d_childNumMat_compact[parentSmallIdx] += 1;
				if (d_visited[parentSmallIdx])
					exitflag = true;
				d_visited[parentSmallIdx] = 1;
			}
			__threadfence();
			if (queueloop) atomicExch(lockArr + parentMap, 0);
		} while (!queueloop);


		smallIdx = parentSmallIdx;
	}
}


__global__ void calcFarLeafKernelPreprocess(int* d_decompress, int* d_farLeafIdx, float* d_farLeafDist, int* d_farLeafColor, int* queue, int seedNum)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	//if (idx == 0)
	//	printf("SeedNum in calcFarLeaf: %d\n", seedNum);
	if (idx >= seedNum) return;
	int smallIdx = queue[idx];
	int index = d_decompress[smallIdx];
	d_farLeafDist[smallIdx] = 0;
	d_farLeafIdx[smallIdx] = smallIdx;
	//color要在运行之后才知道，会不会有问题？改了
	d_farLeafColor[smallIdx] = idx; //color即为seed在seedArr中的序号
}

/*
函数:calcFarLeafKernel
功能:根据追踪得到的parent信息，需要将整个追踪结果划分为不重叠的分支。
其中，需要对追踪结果上的每个节点计算距离它最近的叶子节点（称为最近叶子）。之后，将所有最近叶子相同的节点划分为同一分支。
本kernel用于计算最近叶子，并记录最近叶子的下标、距离和颜色（颜色，即由哪个种子扩展而来）。
*/
__global__ void calcFarLeafKernel(uchar * d_imagePtr, int* d_compress, int* d_decompress, volatile int* d_childNumMat_compact, int* d_parentMat_compact, volatile int* d_farLeafIdx, volatile float* d_farLeafDist, volatile int* d_farLeafColor, int width, int height, int slice, int* queue, int seedNum, int* lockArr, int lockArrSize)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	//if (idx == 0)
	//	printf("SeedNum in calcFarLeaf: %d\n", seedNum);
	if (idx >= seedNum) return;


	int smallIdx = queue[idx];
	int fullIdx = d_decompress[smallIdx];
	float value;
	int parentIdx;
	int parentSmallIdx;
	float parentValue;
	float EuclidDist, tempDist;

	float curDist;
	int curColor;
	int curLeaf;
	int parentLeaf;
	float parentDist;
	int parentColor;

	//以下的两个数组虽然是可变的，但是同时不会有两个thread访问同一个值（这是seed!)
	//实际上curDist应该是0,curLeaf应该是某个seed的Index

	//d_farLeafDist[index] = 0;
	//d_farLeafIdx[index] = index;
	////color要在运行之后才知道，会不会有问题？ 改了
	//d_farLeafColor[index] = idx; //color即为seed在seedArr中的序号

	curDist = 0;// d_farLeafDist[index];
	curLeaf = smallIdx;// d_farLeafIdx[index];
	curColor = idx;//  d_farLeafColor[index];
	volatile bool exitflag = false;
	//__threadfence_system();
	//__syncthreads();

	while (d_parentMat_compact[smallIdx] != smallIdx && (!exitflag))
	{
		//以下的几个数组都是不变的,而且value的值可以优化

		if (d_parentMat_compact[smallIdx] == -1)
		{
			printf("d_parentMat become -1 in farLeafCalc()\n");
		}

		value = d_imagePtr[fullIdx];
		parentSmallIdx = d_parentMat_compact[smallIdx];
		parentIdx = d_decompress[parentSmallIdx];
		parentValue = d_imagePtr[parentIdx];

		int parentMap = parentIdx % lockArrSize;

		//Commented by jifaley 20210816
		//此处的EuclidDist没有专门处理dist26的情况，可能让dist26的情况出错

		int currentZ = fullIdx / (width * height);
		int currentY = fullIdx % (width * height) / width;
		int currentX = fullIdx % width;

		int parentZ = parentIdx / (width * height);
		int parentY = parentIdx % (width * height) / width;
		int parentX = parentIdx % width;

		EuclidDist = sqrtf((currentX - parentX) * (currentX - parentX) + (currentY - parentY) * (currentY - parentY)
			+ (currentZ - parentZ) * (currentZ - parentZ));

		//tempDist = gwdtFunc_gpu_2(EuclidDist, value, parentValue);//or: ==1? //Modified 20210521: 改为像素个数
		//tempDist = 1;
		tempDist = EuclidDist;


		int queueloop;
		volatile int parentChildNum = 0;

		//__threadfence(); //400后出错
		//__syncthreads();

		queueloop = 0;
		do {
			if (queueloop = atomicCAS(lockArr + parentMap, 0, 1) == 0)
			{
				//以下的三个数组都是可变的，因此放在Atomic里面
				parentDist = d_farLeafDist[parentSmallIdx];
				parentLeaf = d_farLeafIdx[parentSmallIdx];
				parentColor = d_farLeafColor[parentSmallIdx];
				//此时可能还是初始状态(-1?)
				parentChildNum = d_childNumMat_compact[parentSmallIdx];



				if (tempDist + curDist >= parentDist + 0.0001f)
				{
					d_farLeafDist[parentSmallIdx] = tempDist + curDist;
					d_farLeafIdx[parentSmallIdx] = curLeaf;
					d_farLeafColor[parentSmallIdx] = curColor;
					parentDist = tempDist + curDist;
					parentLeaf = curLeaf;
					parentColor = curColor;

					//如果更新了，parentDist和parentColor也要保持最新的
				}
				//被访问过一次，childNum就减去1
				parentChildNum--;
				d_childNumMat_compact[parentSmallIdx] = parentChildNum;

				//printf("Idx: %d index: %d  parentIdx: %d  parentDist: %f parentColor: %d pChildNum: %d\n", idx, index, parentIdx, parentDist, parentColor, parentChildNum);

				if (parentChildNum != d_childNumMat_compact[parentSmallIdx])
					printf("parentChildNum sync Error in atomic!\n");

				if (parentChildNum < 0)
				{
					printf("ParentChildNum is negative!\n");
				}

				if (parentChildNum != 0)
					exitflag = true;
				//没有儿子了，最后一个thread接手这个点，带着这个点的（不一定是这个thread本身的）farLeaf 和 farLeafIdx向上走
				else
				{
					smallIdx = parentSmallIdx;
					fullIdx = d_decompress[smallIdx];
					//fullIdx = parentIdx;
					curDist = parentDist;
					curLeaf = parentLeaf;
					curColor = parentColor;
				}
			}
			__threadfence();
			if (queueloop) atomicExch(lockArr + parentMap, 0);
		} while (!queueloop);


		/*if (parentChildNum != d_tempChildNumMat[parentIdx])
			printf("parentChildNum sync Error!\n");*/
			//如果不为0，直接结束thread，说明后面还有其他儿子。

			//__threadfence();
	}


}


/*
函数:constructSegmentKernel
功能:根据追踪得到的parent信息，将整个追踪结果划分为不重叠的分支。其中，需要对追踪结果上的每个节点计算距离它最近的
叶子节点（称为最近叶子）。之后，将所有最近叶子相同的节点划分为同一分支。
*/
__global__ void constructSegmentKernel(uchar * d_imagePtr, int* d_compress, int* d_decompress, int* d_parentMat_compact, int* d_childNumMat_compact, int* d_farLeafIdx, float* d_farLeafDist, int* d_farLeafColor, int width, int height, int slice, int* d_queue, int seedNum,
	int* d_segment_leafIdx, int* d_segment_rootIdx, int* d_segment_length, int* d_segment_parent, float* d_segment_score)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	//int seedNum = *d_queueHead;

	/*if (idx == 0)
		printf("seedNum: %d\n", seedNum);*/
	if (idx >= seedNum) return;

	int leafSmallIdx = d_queue[idx];

	int leafIdx = d_decompress[leafSmallIdx];
	int rootIdx = leafIdx; //本segment的最高祖先
	int rootSmallIdx = d_compress[rootIdx];
	int parentSmallIdx = d_parentMat_compact[rootSmallIdx];
	int parentIdx = d_decompress[parentSmallIdx];

	int level = 1;
	//Modified by jifaley 20211212 from 0 to 1
	int length = 1;

	int fullIdx = leafIdx;
	int smallIdx = d_compress[fullIdx];
	while (parentSmallIdx != smallIdx && parentSmallIdx != -1 && d_farLeafIdx[parentSmallIdx] == leafSmallIdx)
	{
		if (d_childNumMat_compact[smallIdx] >= 2) level++;

		smallIdx = parentSmallIdx;
		fullIdx = d_decompress[smallIdx];

		parentSmallIdx = d_parentMat_compact[smallIdx];
		parentIdx = d_decompress[parentSmallIdx];
		length++;
	}

	rootIdx = fullIdx;
	rootSmallIdx = smallIdx;


	float dst = d_farLeafDist[rootSmallIdx];
	//dst: 本叶子的最先的祖先到叶子的距离。注意如果距离parent最远的叶子不是本叶子，也不再向上回溯。

	d_segment_leafIdx[idx] = leafIdx;
	d_segment_rootIdx[idx] = rootIdx;
	d_segment_length[idx] = length;
	d_segment_score[idx] = dst;

	if (d_parentMat_compact[parentSmallIdx] == parentSmallIdx)
		d_segment_parent[idx] = -1;
	//如果已经找到根节点了
	else
	{
		int leaf2SmallIdx = d_farLeafIdx[parentSmallIdx];
		//否则，一定是祖先有更远的叶子。用leaf2表示。
		if (d_farLeafColor[parentSmallIdx] != d_farLeafColor[leaf2SmallIdx])
			printf("FarLeafColor calc Error!\n");

		d_segment_parent[idx] = d_farLeafColor[leaf2SmallIdx];//d_indexLeaforderMap[leaf2Idx];
		//这样的话，对于这个segment，让他合并到leaf2的分支里面。
	}
}

void constructSegment(std::vector<int>& leafArr, int width, int height, int slice, int newSize, uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress, int* d_parentMat_compact, uchar*  d_statusMat_compact, int* d_childNumMat_compact,
	int*& d_segment_leafIdx, int*& d_segment_rootIdx, int*& d_segment_length, int*& d_segment_parent, float*& d_segment_score, int& segNumber, int darkLeafThreshold)
{
	const int lockArrSize = 10007;
	int* d_lockArr;
	cudaMalloc(&d_lockArr, sizeof(int) * lockArrSize);
	cudaMemset(d_lockArr, 0, sizeof(int) * lockArrSize);

	//01 查找所有的Leaf
	const int queueSize = 2100000;
	int* d_queue;
	cudaMalloc(&d_queue, sizeof(int) * queueSize);
	int* d_queueHead;
	int* d_queueLock;
	cudaMalloc(&d_queueHead, sizeof(int));
	cudaMalloc(&d_queueLock, sizeof(int));
	cudaMemset(d_queueHead, 0, sizeof(int));
	cudaMemset(d_queueLock, 0, sizeof(int));
	cudaDeviceSynchronize();
	findLeafLocalQueueKernel << <(newSize - 1) / 512 + 1, 512 >> > (d_imagePtr, d_decompress, d_statusMat_compact, d_childNumMat_compact, width, height, slice, newSize, d_queue, d_queueHead, d_queueLock, queueSize);

	int* qSize = (int*)malloc(sizeof(int));
	cudaMemcpy(qSize, d_queueHead, sizeof(int), cudaMemcpyDeviceToHost);
	leafArr.resize(*qSize);
	cudaMemcpy(&(leafArr[0]), d_queue, sizeof(int) * (*qSize), cudaMemcpyDeviceToHost);
	int leafNum = *qSize;
	cudaDeviceSynchronize();

	//02 dark Leaf Pruning, 将过暗的叶子节点删除，并重复这个动作，直到遇到分叉点或者足够亮的节点
	findDarkLeafKernel << < 30, 1024 >> > (d_imagePtr_compact, d_compress, d_decompress, d_statusMat_compact, d_childNumMat_compact, d_parentMat_compact, d_queue, d_queueHead, darkLeafThreshold);
	cudaDeviceSynchronize();
	//输出比一下leaf的数量验证正确性	
	cudaMemset(d_lockArr, 0, sizeof(int) * lockArrSize);

	pruneDarkLeafKernel << <  (newSize - 1) / 256 + 1, 256 >> > (d_imagePtr_compact, d_compress, d_decompress, newSize, d_statusMat_compact, d_childNumMat_compact, d_parentMat_compact, darkLeafThreshold, d_lockArr, lockArrSize);
	//cudaDeviceSynchronize();

	//cudaMemset(d_childNumMat_compact, 0, sizeof(int) * newSize);
	//calcChildKernel << <(newSize - 1) / 256 + 1, 256 >> > (d_compress, d_decompress, d_parentMat_compact, d_childNumMat_compact, d_statusMat_compact, width, height, slice, newSize);
	//03 重新统计所有leaf
	cudaMemset(d_queueHead, 0, sizeof(int));
	cudaMemset(d_queueLock, 0, sizeof(int));
	findLeafLocalQueueKernel << <(newSize - 1) / 512 + 1, 512 >> > (d_imagePtr, d_decompress, d_statusMat_compact, d_childNumMat_compact, width, height, slice, newSize, d_queue, d_queueHead, d_queueLock, queueSize);


	cudaMemcpy(qSize, d_queueHead, sizeof(int), cudaMemcpyDeviceToHost);
	leafArr.resize(*qSize);
	cudaMemcpy(&(leafArr[0]), d_queue, sizeof(int) * (*qSize), cudaMemcpyDeviceToHost);
	leafNum = *qSize;

	std::cerr << "Number of Leaves: " << leafNum << std::endl;

	uchar* d_visited;
	cudaMalloc(&d_visited, sizeof(uchar) * newSize);
	cudaMemset(d_visited, 0, sizeof(uchar) * newSize);

	//04 构建所有Segment
	//计算每个点到达的最远的叶子的Index，还有具体的距离


	cudaMemset(d_childNumMat_compact, 0, sizeof(int) * newSize);

	childNumRenewKernel_faster << < (leafNum - 1) / 256 + 1, 256 >> > (d_imagePtr, d_compress, d_decompress, d_childNumMat_compact, d_parentMat_compact, d_visited, d_queue, leafNum, d_lockArr, lockArrSize);
	cudaDeviceSynchronize();

	cudaFree(d_visited);

	int* d_farLeafIdx;
	int* d_farLeafColor;
	float* d_farLeafDist;
	cudaMalloc(&d_farLeafIdx, sizeof(int) * newSize);
	cudaMalloc(&d_farLeafColor, sizeof(int) * newSize);
	cudaMalloc(&d_farLeafDist, sizeof(float) * newSize);

	cudaMemset(d_lockArr, 0, sizeof(int) * lockArrSize);
	cudaMemset(d_farLeafIdx, 0, sizeof(int) * newSize);
	cudaMemset(d_farLeafColor, 0xff, sizeof(int) * newSize);
	thrust::fill(thrust::device, d_farLeafDist, d_farLeafDist + newSize, 0.0f);

	calcFarLeafKernelPreprocess << <(leafNum - 1) / 256 + 1, 256 >> > (d_decompress, d_farLeafIdx, d_farLeafDist, d_farLeafColor, d_queue, leafNum);
	calcFarLeafKernel << <(leafNum - 1) / 256 + 1, 256 >> > (d_imagePtr, d_compress, d_decompress, d_childNumMat_compact, d_parentMat_compact, d_farLeafIdx, d_farLeafDist, d_farLeafColor, width, height, slice, d_queue, leafNum, d_lockArr, lockArrSize);

	//根据上面算出的index和距离层次性计算segment
	//初始情况下每个leaf对应一个segment
	segNumber = leafNum;



	cudaMalloc(&d_segment_leafIdx, sizeof(int) * segNumber);
	cudaMalloc(&d_segment_rootIdx, sizeof(int) * segNumber);
	cudaMalloc(&d_segment_length, sizeof(int) * segNumber);
	cudaMalloc(&d_segment_parent, sizeof(int) * segNumber);
	cudaMalloc(&d_segment_score, sizeof(float) * segNumber);



	constructSegmentKernel << <(segNumber - 1) / 256 + 1, 256 >> >
		(d_imagePtr, d_compress, d_decompress, d_parentMat_compact, d_childNumMat_compact, d_farLeafIdx, d_farLeafDist, d_farLeafColor, width, height, slice, d_queue, leafNum,
			d_segment_leafIdx, d_segment_rootIdx, d_segment_length, d_segment_parent, d_segment_score);

	cudaFree(d_farLeafDist);
	cudaFree(d_farLeafIdx);
	cudaFree(d_farLeafColor);
	cudaFree(d_queue);
	cudaFree(d_queueHead);
	cudaFree(d_queueLock);
	cudaFree(d_lockArr);
	free(qSize);
}

__device__ int getfather_gpu_2(int* d_disjointSet, int x)
{
	if (d_disjointSet[x] == x) return x;
	return d_disjointSet[x] = getfather_gpu_2(d_disjointSet, d_disjointSet[x]);
}

__global__ void getMaxLengthColorKernel(int segNumber, int* d_segment_length, uchar* d_segmentKeep, float scoreThreshold, int* d_segment_rootIdx, short int* d_seedNumberPtr, int* d_compress, int* d_disjointSet, float* colorCount)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= segNumber) return;
	int curLength = d_segment_length[idx];

	int curPos = d_segment_rootIdx[idx];
	int curSmallIdx = d_compress[curPos];

	int seed = d_seedNumberPtr[curSmallIdx];
	int color = getfather_gpu_2(d_disjointSet, seed);

	atomicAdd(colorCount + color, (float)curLength);
}



__global__ void scoreSegmentFilterKernel(int segNumber, float* d_segment_score, uchar* d_segmentKeep, float scoreThreshold, int* d_segment_rootIdx, short int* d_seedNumberPtr, int* d_compress, int* d_disjointSet, int centerColor)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= segNumber) return;
	int curLength = d_segment_score[idx];

	if (curLength >= scoreThreshold)
	{
		d_segmentKeep[idx] = 1; //1:keep 0:prune


#ifdef __UNIQUE__COLOR
		int curPos = d_segment_rootIdx[idx];
		int curSmallIdx = d_compress[curPos];

		int seed = d_seedNumberPtr[curSmallIdx];
		int color = getfather_gpu_2(d_disjointSet, seed);

		if (color != centerColor)
		{
			//此处用于控制是否只保留最核心的部分，而丢弃其他颜色部分
			d_segmentKeep[idx] = 0;
		}
#endif  //!__UNIQUE__COLOR

	}

	else
	{
		d_segmentKeep[idx] = 0; //filterByLengthCount++;
	}
}


template<typename T>
struct is_zero {
	__host__ __device__
		bool operator()(T x) const
	{
		return x == 0;
	}
};


using thrust::placeholders::_1;

__global__
void getCompressMap_segment(int* d_compress, int* d_decompress, int* d_segment_leafIdx, int* d_segment_rootIdx, int* d_segment_length, int* d_segment_parent,
	int* d_segment_leafIdx_filtered, int* d_segment_rootIdx_filtered, int* d_segment_length_filtered, int* d_segment_parent_filtered, int segNumberFiltered)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= segNumberFiltered) return;
	int fullIdx = d_decompress[smallIdx];

	d_compress[fullIdx] = smallIdx;
	d_segment_leafIdx_filtered[smallIdx] = d_segment_leafIdx[fullIdx];
	d_segment_rootIdx_filtered[smallIdx] = d_segment_rootIdx[fullIdx];
	d_segment_length_filtered[smallIdx] = d_segment_length[fullIdx];

	int parentIdx = d_segment_parent[fullIdx];
	if (parentIdx == -1)
	{
		d_segment_parent_filtered[smallIdx] = -1;
	}
	else
	{
		d_segment_parent_filtered[smallIdx] = d_compress[parentIdx];
		//parentIdx 存在时，它映射过去的结果也可能是-1。不过，我们把d_compress的初值设置为-1了
	}
}


void filterSegment(int* d_segment_leafIdx, int* d_segment_rootIdx, int* d_segment_length, int* d_segment_parent, float* d_segment_score,  short int* d_seedNumberPtr, int* d_disjointSet,  int* d_compress_outer, int totalColor, int scoreThreshold, int segNumber, int& segNumberFiltered)
{

	cudaError_t errorCheck;

	//int scoreThreshold = lengthThreshold;

	int* d_compress;
	int* d_decompress;
	int* d_sequence;
	uchar* d_segmentKeep;

	float* d_colorCount;
	float* h_colorCount = (float*)malloc(sizeof(float) * totalColor);

	cudaMalloc(&d_colorCount, sizeof(float) * totalColor);
	thrust::fill(thrust::device, d_colorCount, d_colorCount + totalColor, 0.0f);

	cudaMalloc(&d_segmentKeep, sizeof(uchar) * segNumber);
	cudaMemset(d_segmentKeep, 0, sizeof(uchar) * segNumber);

	getMaxLengthColorKernel << <(segNumber - 1) / 256 + 1, 256 >> > (segNumber, d_segment_length, d_segmentKeep, scoreThreshold, d_segment_rootIdx, d_seedNumberPtr, d_compress_outer, d_disjointSet, d_colorCount);

	cudaMemcpy(h_colorCount, d_colorCount, sizeof(float) * totalColor, cudaMemcpyDeviceToHost);

	float maxlength = 0;
	int centerColor = -1;
	for (int i = 1; i < totalColor; i++)
	{
		if (h_colorCount[i] > maxlength)
		{
			maxlength = h_colorCount[i];
			centerColor = i;
		}
		if (h_colorCount[i] > 1)
		{
			//printf("seed: %d, length: %f \n", i, h_colorCount[i]);
		}
	}

	scoreSegmentFilterKernel << <(segNumber - 1) / 256 + 1, 256 >> > (segNumber, d_segment_score, d_segmentKeep, scoreThreshold, d_segment_rootIdx, d_seedNumberPtr, d_compress_outer, d_disjointSet, centerColor);


	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "Duing Sore filter " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}


	cudaMalloc(&d_compress, sizeof(int) * segNumber);
	cudaMalloc(&d_sequence, sizeof(int) * segNumber);
	cudaMemset(d_compress, 0xff, sizeof(int) * segNumber);

	//经过copy_if后，d_sequence中留下的是原始体数据非0值的下标。该操作是stable的。 newSize即为非0值的个数。
	try
	{
		int* d_copy_end = thrust::copy_if(thrust::device, thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(segNumber), d_segmentKeep, d_sequence, _1 != 0);
		segNumberFiltered = d_copy_end - d_sequence;
	}
	catch (thrust::system_error error)
	{
		std::cerr << std::string(error.what()) << std::endl;
	}

	std::cerr << "Before: " << segNumber << " After: " << segNumberFiltered << std::endl;

	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "Duing copy_If " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	cudaMalloc(&d_decompress, sizeof(int) * segNumberFiltered);
	cudaMemcpy(d_decompress, d_sequence, sizeof(int) * segNumberFiltered, cudaMemcpyDeviceToDevice);

	int* d_segment_leafIdx_filtered;
	int* d_segment_rootIdx_filtered;
	int* d_segment_length_filtered;
	int* d_segment_parent_filtered;

	cudaMalloc(&d_segment_leafIdx_filtered, sizeof(int) * segNumberFiltered);
	cudaMalloc(&d_segment_rootIdx_filtered, sizeof(int) * segNumberFiltered);
	cudaMalloc(&d_segment_length_filtered, sizeof(int) * segNumberFiltered);
	cudaMalloc(&d_segment_parent_filtered, sizeof(int) * segNumberFiltered);


	//计算对应的映射
	getCompressMap_segment << < (segNumberFiltered - 1) / 256 + 1, 256 >> > (d_compress, d_decompress, d_segment_leafIdx, d_segment_rootIdx, d_segment_length, d_segment_parent,
		d_segment_leafIdx_filtered, d_segment_rootIdx_filtered, d_segment_length_filtered, d_segment_parent_filtered, segNumberFiltered);

	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "Duing get compress map " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}



	cudaMemcpy(d_segment_leafIdx, d_segment_leafIdx_filtered, sizeof(int) * segNumberFiltered, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_segment_rootIdx, d_segment_rootIdx_filtered, sizeof(int) * segNumberFiltered, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_segment_length, d_segment_length_filtered, sizeof(int) * segNumberFiltered, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_segment_parent, d_segment_parent_filtered, sizeof(int) * segNumberFiltered, cudaMemcpyDeviceToDevice);

	cudaFree(d_segmentKeep);
	cudaFree(d_sequence);
	cudaFree(d_segment_leafIdx_filtered);
	cudaFree(d_segment_rootIdx_filtered);
	cudaFree(d_segment_length_filtered);
	cudaFree(d_segment_parent_filtered);
	cudaFree(d_compress);
	cudaFree(d_decompress);
	cudaFree(d_colorCount);
	free(h_colorCount);
}



__global__ void travelSegmentKernelFinal(int* d_compress, int* d_decompress, int* d_parentMat_compact, int* d_segment_leafIdx, int* d_segment_rootIdx, int* d_swcIdxToIndex, int* d_isLeaf, int* d_isRoot, int* d_indexToSwcIndex, int segNumberFinal)
{
	//本Kernel显然可以优化，每个block处理一个分支放在smem里面之类的，length都确定了,可以用idx计算具体offset（刚才已经算了）
	int pointCounter = 0;
	int index, parentIdx, leafIdx, rootIdx, smallIdx;

	for (int i = 0; i < segNumberFinal; i++)
	{
		leafIdx = d_segment_leafIdx[i];
		rootIdx = d_segment_rootIdx[i];
		index = leafIdx;
		smallIdx = d_compress[index];

		while (1)
		{
			if (d_indexToSwcIndex[index] == -1)
			{
				//swc的首位是-1，所以映射全部+1
				d_indexToSwcIndex[index] = pointCounter + 1;
			}

			if (index == leafIdx) d_isLeaf[pointCounter] = 1;
			if (index == rootIdx) d_isRoot[pointCounter] = 1;
			d_swcIdxToIndex[pointCounter++] = index;
			if (index == rootIdx) break;
			smallIdx = d_parentMat_compact[smallIdx];
			index = d_decompress[smallIdx];
		}
	}

	printf("Point Counter in travelSegmentKernelFinal: %d\n", pointCounter);
}





__global__ void getColorKernel(int* d_compress, int* d_color, int * d_swcIdxArr, short int* d_seedNumberPtr, int* d_disjointSet, int vSwcSize)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= vSwcSize) return;

	//注意！在SWC文件格式中，下标是从1开始的
	if (idx == 0) return;


	int curPos = d_swcIdxArr[idx];
	int curSmallIdx = d_compress[curPos];

	int seed = d_seedNumberPtr[curSmallIdx];
	int color = getfather_gpu_2(d_disjointSet, seed);
	d_color[idx] = color;

}


__global__ void getParentSwcIdxAndRadiusKernel(int* d_compress, int* d_decompress, int* d_parentMat_compact, uchar* d_radiusMat_compact, int* d_segment_leafIdx, int* d_segment_rootIdx, int* d_pointIdxMat, int* d_isLeaf, int* d_isRoot, int* d_indexToSwcIndex, int* d_parentSwcIdx, uchar* d_radiusSwc, int segNumberFiltered, int total_count)
{

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= total_count) return;
	int index = d_pointIdxMat[idx];
	//d_radiusSwc[idx] = d_radiusMat[index];

	//int parentIdx = d_parentMat[index];
	int smallIdx = d_compress[index];

	d_radiusSwc[idx] = d_radiusMat_compact[smallIdx];
	int parentSmallIdx = d_parentMat_compact[smallIdx];
	int parentIdx;
	if (parentSmallIdx != -1)
		parentIdx = d_decompress[parentSmallIdx];
	else
		parentIdx = -1;


	int parentSwcIdx = -1;

	if (parentIdx != index && parentIdx != -1)
		parentSwcIdx = d_indexToSwcIndex[parentIdx];
	d_parentSwcIdx[idx] = parentSwcIdx;
}

bool smooth_curve_and_radius(std::vector<swcPoint*> & mCoord, int winsize);

void smooth(std::vector<swcPoint> & vSwcPoint, std::vector<int> & vIsLeaf, std::vector<int> & vIsRoot, int winsize);


void outputSwc(int* d_compress, int* d_decompress, int* d_parentMat_compact, uchar* d_radiusMat_compact, short int* d_seedNumberPtr, int* d_disjointSet, std::vector<int>& segment_leafIdx_final, std::vector<int>& segment_rootIdx_final, std::vector<int>& segment_length_final, int width, int height, int slice, int* d_segment_leafIdx, int* d_segment_rootIdx, int segNumberFiltered, int segNumberFinal, std::string inputName)
{
	FILE* swc_out = nullptr;

	int last_slash_pos = inputName.find_last_of("\\");

	std::string imageName;

	if (last_slash_pos == std::string::npos)
		imageName = inputName;
	else
		imageName = inputName.substr(last_slash_pos + 1, inputName.length() - last_slash_pos);


	std::string output_path = std::string("results\\") + imageName + "_InstantTrace.swc";

	fopen_s(&swc_out, output_path.c_str(), "w+");


	int maxNumOfSwc = 0;

	std::cerr << "segNumberFiltered: " << segNumberFiltered << " vFilteredSegmentsFinal.size(): " << segNumberFinal << std::endl;
	for (int i = 0; i < segNumberFinal; i++)
	{
		int length = segment_length_final[i];//seg->length;
		int leafIdx = segment_leafIdx_final[i];//seg->leafIdx;
		int rootIdx = segment_rootIdx_final[i];//seg->rootIdx;
		maxNumOfSwc += (length);
	}

	std::cerr << "MaxNumof SWC: " << maxNumOfSwc << std::endl;

	int* d_indexToSwcIndex;
	//注意: swc index 从1开始
	cudaMalloc(&d_indexToSwcIndex, sizeof(int) * width * height * slice);
	cudaMemset(d_indexToSwcIndex, 0xff, sizeof(int) * width * height * slice);

	int* d_swcIdxToIndex;
	cudaMalloc(&d_swcIdxToIndex, sizeof(int) * ((int)maxNumOfSwc + 1));

	int* d_parentSwcIdx;
	cudaMalloc(&d_parentSwcIdx, sizeof(int) * ((int)maxNumOfSwc + 1));

	int* d_isLeaf;
	int* d_isRoot;
	cudaMalloc(&d_isLeaf, sizeof(int) * ((int)maxNumOfSwc + 1));
	cudaMalloc(&d_isRoot, sizeof(int) * ((int)maxNumOfSwc + 1));
	cudaMemset(d_isLeaf, 0, sizeof(int) * ((int)maxNumOfSwc + 1));
	cudaMemset(d_isRoot, 0, sizeof(int) * ((int)maxNumOfSwc + 1));


	//这两个应该更新，之前的值覆盖掉不要了
	cudaMemcpy(d_segment_leafIdx, &segment_leafIdx_final[0], sizeof(int) * segNumberFinal, cudaMemcpyHostToDevice);
	cudaMemcpy(d_segment_rootIdx, &segment_rootIdx_final[0], sizeof(int) * segNumberFinal, cudaMemcpyHostToDevice);

	travelSegmentKernelFinal << <1, 1 >> > (d_compress, d_decompress, d_parentMat_compact, d_segment_leafIdx, d_segment_rootIdx, d_swcIdxToIndex, d_isLeaf, d_isRoot, d_indexToSwcIndex, segNumberFinal);


	std::vector<int> vSwcIndex(maxNumOfSwc + 1, 0);
	std::vector<int> vIsLeaf(maxNumOfSwc + 1, 0);
	std::vector<int> vIsRoot(maxNumOfSwc + 1, 0);
	vSwcIndex[0] = -1;
	vIsLeaf[0] = 0;
	vIsRoot[0] = 0;

	cudaMemcpy(&vIsLeaf[1], d_isLeaf, sizeof(int) * maxNumOfSwc, cudaMemcpyDeviceToHost);
	cudaMemcpy(&vIsRoot[1], d_isRoot, sizeof(int) * maxNumOfSwc, cudaMemcpyDeviceToHost);
	cudaMemcpy(&vSwcIndex[1], d_swcIdxToIndex, sizeof(int) * maxNumOfSwc, cudaMemcpyDeviceToHost);
	int numberOfSwc = 0;
	int multicounter = 0;



	int vSwcSize = vSwcIndex.size();
	std::cerr << "vSwcSize " << vSwcIndex.size() << " numofSWc " << numberOfSwc << std::endl;

	int* d_color;
	cudaMalloc(&d_color, sizeof(int) * vSwcSize);
	int* d_swcIdxArr;
	cudaMalloc(&d_swcIdxArr, sizeof(int) * vSwcSize);

	cudaMemcpy(d_swcIdxArr, &(vSwcIndex[0]), sizeof(int) * vSwcSize, cudaMemcpyHostToDevice);
	int* colorMat = (int*)malloc(sizeof(int) * vSwcSize);


	getColorKernel << <(vSwcSize - 1) / 256 + 1, 256 >> > (d_compress, d_color, d_swcIdxArr, d_seedNumberPtr, d_disjointSet, vSwcSize);

	uchar* d_radiusSwc;
	cudaMalloc(&d_radiusSwc, sizeof(uchar)* maxNumOfSwc);
	uchar* radiusSwc = (uchar*)malloc(sizeof(uchar) * (maxNumOfSwc + 1));

	getParentSwcIdxAndRadiusKernel << <(maxNumOfSwc - 1) / 256 + 1, 256 >> > (d_compress, d_decompress, d_parentMat_compact, d_radiusMat_compact, d_segment_leafIdx, d_segment_rootIdx, d_swcIdxToIndex, d_isLeaf, d_isRoot, d_indexToSwcIndex, d_parentSwcIdx, d_radiusSwc, segNumberFinal, maxNumOfSwc);

	int * h_parentSwcIdx = (int*)malloc(sizeof(int) * (maxNumOfSwc + 1));
	cudaMemcpy(h_parentSwcIdx + 1, d_parentSwcIdx, sizeof(int) * maxNumOfSwc, cudaMemcpyDeviceToHost);
	cudaMemcpy(colorMat, d_color, sizeof(int) * vSwcSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(radiusSwc + 1, d_radiusSwc, sizeof(uchar) * maxNumOfSwc, cudaMemcpyDeviceToHost);


	cudaFree(d_color);
	cudaFree(d_swcIdxArr);

	cudaFree(d_indexToSwcIndex);
	cudaFree(d_isLeaf);
	cudaFree(d_isRoot);

	cudaFree(d_swcIdxToIndex);
	cudaFree(d_radiusSwc);



	vector<swcPoint> vSwcPoint(vSwcIndex.size());

	for (int i = 1; i < vSwcIndex.size(); i++)
	{
		int index = vSwcIndex[i];
		int parentSwcIndex = h_parentSwcIdx[i];
		int z = index / (width * height);
		int y = (index % (width * height)) / width;
		int x = index % width;
		int r = radiusSwc[i];

		//Adding Pruning Merge 20211030
		vSwcPoint[i] = swcPoint(x, height - y, z, r, i, parentSwcIndex, colorMat[i]);//disjointSet[d_seedNumberPtr[index]]);
		//fprintf(swc_out, "%d %d %d %d %d %d %d\n", i, 0, x, height - y, z, r, parentSwcIndex);
		//Ends

	}



	free(colorMat);
	free(radiusSwc);
	free(h_parentSwcIdx);


	smooth(vSwcPoint, vIsLeaf, vIsRoot, 5);


	for (int i = 1; i < vSwcIndex.size(); i++)
	{
		swcPoint* it = &vSwcPoint[i];
		//Adding Pruning Merge 20211030
		//fprintf(swc_out, "%d %d %d %d %d %d %d\n", it->swcIndex, 0, it->x, it->y, it->z, it->r, it->parentSwcIndex);

		int this_color = ((it->seedNumber - 1) % 12 + 2);

		if (it->seedNumber == 1)
			this_color = 7;
		if (it->seedNumber == 2)
			this_color = 5;

		fprintf(swc_out, "%d %d %d %d %d %d %d\n", it->swcIndex, this_color, it->x, it->y, it->z, it->r, it->parentSwcIndex);
		//Ends
	}

	fclose(swc_out);


}




bool smooth_curve_and_radius(std::vector<swcPoint *> & mCoord, int winsize)
{
	//std::cout<<" smooth_curve ";
	if (winsize < 2) return true;

	if (mCoord.size() < winsize * 2 + 1)
		winsize = (mCoord.size() - 1) / 2;

	if (winsize < 2) return true;

	std::vector<swcPoint*> mC = mCoord; // a copy
	int N = mCoord.size();
	int halfwin = winsize / 2;

	for (int i = 1; i < N - 1; i++) // don't move start & end point
	{
		std::vector<swcPoint*> winC;
		std::vector<double> winW;
		winC.clear();
		winW.clear();

		winC.push_back(mC[i]);
		winW.push_back(1. + halfwin);
		for (int j = 1; j <= halfwin; j++)
		{
			int k1 = i + j;   if (k1 < 0) k1 = 0;  if (k1 > N - 1) k1 = N - 1;
			int k2 = i - j;   if (k2 < 0) k2 = 0;  if (k2 > N - 1) k2 = N - 1;
			winC.push_back(mC[k1]);
			winC.push_back(mC[k2]);
			//winW.push_back(1);
			//winW.push_back(1);
			winW.push_back(1. + halfwin - j);
			winW.push_back(1. + halfwin - j);
		}
		//std::cout<<"winC.size = "<<winC.size()<<"\n";

		double s, x, y, z, r;
		s = x = y = z = 0;
		for (int ii = 0; ii < winC.size(); ii++)
		{
			x += winW[ii] * winC[ii]->x;
			y += winW[ii] * winC[ii]->y;
			z += winW[ii] * winC[ii]->z;
			r += winW[ii] * winC[ii]->r;
			s += winW[ii];
		}
		if (s)
		{
			x /= s;
			y /= s;
			z /= s;
			r /= s;
		}

		//mCoord[i]->x = x; // output
		//mCoord[i]->y = y; // output
		//mCoord[i]->z = z; // output
		mCoord[i]->r = r; // output
	}
	return true;
}

void smooth(std::vector<swcPoint> & vSwcPoint, std::vector<int> & vIsLeaf, std::vector<int> & vIsRoot, int winsize)
{
	std::vector<bool>visited(vSwcPoint.size(), false);

	for (int i = 1; i < vSwcPoint.size(); i++)
	{
		if (visited[i] || !vIsLeaf[i]) continue;
		swcPoint* leaf_marker = &vSwcPoint[i];
		std::vector<swcPoint*> seg_markers;
		swcPoint * p = leaf_marker;
		while (1)
		{
			seg_markers.push_back(p);
			p = &vSwcPoint[p->parentSwcIndex];
			if (vIsRoot[p->swcIndex])
				break;
		}
		//if (vIsRoot[p->swcIndex])
		//	std::cerr << p->swcIndex << std::endl;
		smooth_curve_and_radius(seg_markers, 10);
	}
}
