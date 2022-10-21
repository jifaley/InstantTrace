//#define  CUDA_API_PER_THREAD_DEFAULT_STREAM
#pragma warning (disable:4819)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.h"
#include "poissonSample.h"
#include "threshold.h"
#include "pruning.h"
#include "fastmarching.h"
#include "mergeSegments.h"
#include "compaction.h"

#define __USE__DIST26

//If one wants to try the version without multi seeds, use this option
//#define __ONLY__ONE__SEED

//Warmup:给GPU预热时间
//Kernel for GPU warmup
__global__ void warmupKernel()
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}


dim3 find_max(dim3 target, int width, int height, int slice, uchar* arr)
{
	dim3 result = target;
	int max_intensity = arr[target.z * width * height + target.y * width + target.x];

	for (int k = target.z - 3; k < target.z + 3; k++)
	{
		if (k < 0 || k >= slice)
			continue;
		for (int i = target.y - 3; i < target.y + 3; i++)
		{
			if (i < 0 || i >= height)
				continue;
			for (int j = target.x - 3; j < target.x + 3; j++)
			{
				if (j < 0 || j >= width)
					continue;
				int cur_intensity = arr[k * width * height + i * width + j];
				if (cur_intensity > max_intensity)
				{
					max_intensity = cur_intensity;
					result.z = k; result.y = i; result.x = j;
				}
			}
		}
	}
	std::cerr << "Previous: " << target.x << ' ' << target.y << ' ' << target.z << " Intensity: " << (int)(arr[target.z * width * height + target.y * width + target.x])
		<< std::endl;
	std::cerr << "Now: " << result.x << ' ' << result.y << ' ' << result.z << " Intensity: " << max_intensity << std::endl;
	return result;
}


int main(int argc, char* argv[])
{
	TimerClock timer, timer2, timer3;
	cudaError_t errorCheck;

	std::vector<std::string> files;
	std::vector<std::string> names;

	std::string inputPath = "data";

	if (argc > 1)
	{
		inputPath = argv[1];
	}

	if (inputPath.length() > 4 && inputPath.substr(inputPath.length() - 4, 4) == std::string(".tif"))
	{
		files.push_back(inputPath);
		names.push_back(inputPath);
	}
	else
	{
		/*When use multi files*/
		getFiles(inputPath, files, names);
	}

	for (int item = 0; item < names.size(); item++)
	{
		std::string file = files[item];
		std::string name = names[item];

		std::cerr << "file: " << item + 1 << "/" << names.size() << std::endl;
		cudaDeviceReset();

		timer.update();

		//Step 0: 读图, CPU端耗时约600-800ms，读入后拷贝原始体数据到GPU耗时约40ms
		std::cerr << "Loading image.." << std::endl;
		string inputName;

		inputName = file;

		int *imageShape = new int[3];
		uchar* h_imagePtr = loadImage(inputName, imageShape);
		int width = imageShape[0]; //963
		int height = imageShape[1]; //305
		int slice = imageShape[2]; //140
		std::cerr << "Image Reading cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;

		timer.update();
		timer2.update();


		//Step 1: GPU预热
		//Step 1: GPU warmup
		warmupKernel << <1, 1 >> > ();

		printf("warmup cost: %dms\n\n", (int)(timer.getTimerMilliSec()));
		timer.update();
		timer3.update();
		uchar* d_imagePtr;
		cudaMalloc((void**)&d_imagePtr, sizeof(uchar) * width * height * slice);
		//将原图从host复制到device
		cudaMemcpy(d_imagePtr, h_imagePtr, sizeof(uchar) * width * height * slice, cudaMemcpyHostToDevice);
		std::cerr << "first malloc and memcpy cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();


		int g2 = getGlobalThreshold(h_imagePtr, d_imagePtr, width, height, slice);

		std::cerr << "get globalThreshold cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();



		//Step 2: 添加阈值等基础预处理
		//Step 2: Preprocessing, such as thresholding(global/local)
		//依据blockSize分块，根据块内灰度情况自适应添加阈值
		//耗时:约20ms
		int localThresholdBlockSize = 32;


		bool useLocalTh = true;

		//useLocalTh = false; // for flycircuit

		if (argc > 3)
		{

			string ifUseLocal = argv[3];

			if (ifUseLocal == "0")
				useLocalTh = false;
		}


		if (useLocalTh)
			addLocalThreshold(d_imagePtr, width, height, slice, localThresholdBlockSize);
		std::cerr << "add localThreshold cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();


		//添加全局阈值
		//耗时：<1ms
		int globalThreshold = 5;
		globalThreshold = g2;

		if (argc > 2)
		{
			if (atoi(argv[2]) != -1)
				globalThreshold = atoi(argv[2]);
		}

		//globalThreshold = 1;// for flycircuit

		if (globalThreshold > 1)
			addGlobalThreshold(d_imagePtr, width, height, slice, globalThreshold);
		std::cerr << "Add GlobalThreshold cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();

		//对足够亮的区域，将其周围的暗区赋值为1，以填补亮区之间的缝隙
		//耗时: <1ms
		int paddingThreshold = globalThreshold + 2;
		addDarkPadding(d_imagePtr, width, height, slice, paddingThreshold);
		cudaDeviceSynchronize();
		std::cerr << "Add Padding cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();


		//Step 3: 压缩原始图像，除去0值 (称为stream compaction, 流压缩) 
		//Step 3: Stream Compaction, remove the zero-valued elements and compress the original image
		//耗时:约70ms


		//下面两个数组存储了原图和压缩后图之间的映射
		//compress: global Idx-> compressed Idx
		//decompress: compressed Idx-> global Idx
		int* d_compress;
		int* d_decompress;
		uchar* d_imagePtr_compact;

		//这两个指针都在函数里面分配空间,因为d_decompress未知大小
		int newSize = -1;

		compactImage(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, width, height, slice, newSize);
		std::cerr << "Old: " << width * height * slice << " New:" << newSize << std::endl;

		cudaDeviceSynchronize();
		std::cerr << "Stream Compaction cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();


		//Step 4: 计算每个点的控制半径，用于最后的剪枝判断
		//Step 4: Calculating the radius of the voxels (i.e., the radius of the neighborhood in the foreground)  
		//耗时:约70ms
		uchar* d_radiusMat_compact;
		cudaMalloc((void**)&d_radiusMat_compact, sizeof(uchar) * newSize);
		thrust::fill(thrust::device, d_radiusMat_compact, d_radiusMat_compact + newSize, 0);

		calcRadius_gpu_compact(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_radiusMat_compact, width, height, slice, newSize, globalThreshold);


		cudaDeviceSynchronize();
		errorCheck = cudaGetLastError();
		if (errorCheck != cudaSuccess) {
			std::cerr << "During CalcRadius GPU " << cudaGetErrorString(errorCheck) << std::endl;
			system("pause");
			return;
		}


		//Step 5: GrayWeight Distacne Transform, 灰度距离变换，作为预处理的一种，让图像更明显
		//Step 5: GrayWeight Distacne Transform, (See Xiao et al. APP2: automatic tracing of 3d neuron morphology...)
		//As a step of preprocessing, making the neuron more clear
		//耗时:约80ms

		addGreyWeightTransform(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, width, height, slice, newSize);

		std::cerr << "GWDT cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();

		int maxpos = 0;


		//寻找最大值，作为神经元胞体中心
		//Find the center of the neuron soma
		int maxRadius;
		int maxPos;
		
		getCenterPos(d_compress, d_decompress, d_radiusMat_compact, width, height, slice, newSize, maxPos, maxRadius);

		
		int max_pos = maxPos;
		int mz = max_pos / (width * height);
		int my = max_pos % (width * height) / width;
		int mx = max_pos % width;

		dim3 center(mx, my, mz);
		std::cerr << "Neuron Center Pos:" << maxRadius << ' ' << mz << ' ' << my << ' ' << mx << std::endl;
		std::cerr << "Calc Radius GPU cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;




		timer.update();

		std::cerr << "[Breakdown] Total Preprocessing Cost: " << timer3.getTimerMilliSec() << "ms" << std::endl << std::endl;

		timer3.update();


		//Step 6: 泊松采样,给后面的追踪提供种子。实现自Wei et al.的parallel poisson disk sampling
		//Step 6: Parallel Poisson Disk sampling, see (L.-Y. Wei. Parallel poisson disk sampling.) 
		//分为采样和筛选两个步骤，筛选需要传入原数组（足够亮的才要），还有传入胞体中心和它的半径
		//The steps include sampling and filtering. Only the samples with high intensity are accepted.
		//返回的是std::vector seedArr，里面存储了种子点的位置, 同时该数组的最后添加了一个元素，胞体中心，作为额外的也是最重要的种子
		//The output, "seedArr", includes the locations of sampled seed points. Specically, the last seed point is the center of neuron soma.

		std::vector<int> seedArr;

#ifdef __ONLY__ONE__SEED
		seedArr.push_back(center.z * width * height + center.y * width + center.x);
#else
		doPoissonSample2(seedArr, center, maxRadius, width, height, slice, newSize, d_imagePtr, d_imagePtr_compact, d_compress, d_decompress);

#endif // __ONLY__ONE__SEED



		cudaDeviceSynchronize();
		std::cerr << "GPU sampling postprocessing and output cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();

		cudaDeviceSynchronize();
		errorCheck = cudaGetLastError();
		if (errorCheck != cudaSuccess) {
			std::cerr << "During Poisson Sample " << cudaGetErrorString(errorCheck) << std::endl;
			system("pause");
			return;
		}

		std::cerr << "[Breakdown] Total Seed Generating Cost: " << timer3.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer3.update();

		//Step 7: 初始追踪,使用并行的fast-marching算法，或者说并行最短路算法
		//将会从泊松采样提供的种子开始扩展，每个种子扩展出一部分分支。
		//Step 7: Initial neuron tracing, using parallel fast-marching algorithm, or parallel shortest path algorithm.
		//The algorithm will start from the sampled seeds in parallel, and generate multi traced branches.

		short int* d_seedNumberPtr;
		uchar* d_activeMat_compact;
		cudaMalloc((void**)&d_activeMat_compact, sizeof(uchar) * newSize);
		cudaMemset(d_activeMat_compact, FAR, sizeof(uchar) * newSize);

		cudaMalloc((void**)& d_seedNumberPtr, sizeof(short int) * newSize);
		cudaMemset(d_seedNumberPtr, 0, sizeof(short int) * newSize);

		int* d_parentPtr_compact;
		cudaMalloc(&d_parentPtr_compact, sizeof(int) * newSize * 2);
		cudaMemset(d_parentPtr_compact, 0xff, sizeof(int) * newSize * 2);


		int* d_childNumMat;
		cudaMalloc(&d_childNumMat, sizeof(int) * newSize);
		cudaMemset(d_childNumMat, 0, sizeof(int) * newSize);


		buildInitNeuron(seedArr, d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_parentPtr_compact, d_seedNumberPtr, d_activeMat_compact, d_childNumMat, width, height, slice, newSize);


		std::cerr << "Init neuron generating cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();

		std::cerr << "[Breakdown] Total Init Generating Cost: " << timer3.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer3.update();

		//Step 8: 分支合并算法
		//Step 8: Merging the result of initial tracing.
		//耗时:约50ms
		std::vector<int> leafArr;
		int leafCount = 0;
		int validLeafCount = 0;

		//并查集，用于判断不同分支之间是否进行了合并
		//disjoint set for checking the merging status.
		std::vector<int> disjointSet(seedArr.size() + 1, 0);
		for (int it = 0; it < disjointSet.size(); it++)
			disjointSet[it] = it;

		int* d_disjointSet;
		cudaMalloc(&d_disjointSet, sizeof(int) * disjointSet.size());
		cudaMemcpy(d_disjointSet, &(disjointSet[0]), sizeof(int) * disjointSet.size(), cudaMemcpyHostToDevice);

		mergeSegments(seedArr, disjointSet, width, height, slice, newSize, d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_childNumMat, d_radiusMat_compact, d_parentPtr_compact, d_seedNumberPtr, d_disjointSet);

		std::cerr << "MergeSegments cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();

		std::cerr << "[Breakdown] Total Merging Cost: " << timer3.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer3.update();

		cudaDeviceSynchronize();
		errorCheck = cudaGetLastError();
		if (errorCheck != cudaSuccess) {
			std::cerr << "Before Pruning " << cudaGetErrorString(errorCheck) << std::endl;
			system("pause");
			return;
		}


		//Step 9: 最终剪枝. 输出结果为swc格式。
		//Step 9: The final pruning process. The output is in swc format.
		pruneLeaf_3d_gpu(leafArr, validLeafCount, disjointSet, width, height, slice, newSize, d_radiusMat_compact, d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_parentPtr_compact, d_activeMat_compact, d_childNumMat, d_seedNumberPtr, d_disjointSet, inputName);
		//pruning
		std::cerr << "After Pruning" << std::endl;
		std::cerr << validLeafCount << std::endl;
		std::cerr << "Pruning cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();



		cudaFree(d_compress);
		cudaFree(d_decompress);
		cudaFree(d_imagePtr_compact);

		free(h_imagePtr);
		cudaFree(d_imagePtr);
		cudaFree(d_activeMat_compact);
		cudaFree(d_parentPtr_compact);
		cudaFree(d_childNumMat);
		cudaFree(d_disjointSet);
		cudaFree(d_seedNumberPtr);

		cudaFree(d_radiusMat_compact);

		

		std::cerr << "[Breakdown] Total Pruning Cost: " << timer3.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer3.update();


		std::cerr << "Total time cost: " << timer2.getTimerMilliSec() << "ms" << std::endl;
		timer2.update();
	}

	return 0;
}



