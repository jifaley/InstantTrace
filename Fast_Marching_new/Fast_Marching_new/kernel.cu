//#define  CUDA_API_PER_THREAD_DEFAULT_STREAM
#pragma warning (disable:4819)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.h"
#include "histogram.h"
#include "poissonSample.h"
#include "threshold.h"
#include "pruning.h"
#include "fastmarching.h"
#include "mergeSegments.h"
#include "compaction.h"

#define __USE__DIST26
//#define __ONLY__ONE__SEED


__global__ void warmupKernel()
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}


int main(int argc, char* argv[])
{
	TimerClock timer, timer2, timer3;
	cudaError_t errorCheck;

	std::vector<std::string> files;
	std::vector<std::string> names;

	std::string inputPath = "data//bright";

	getFiles(inputPath, files, names);



	for (int item = 0; item < names.size(); item++)
	{
		std::string file = files[item];
		std::string name = names[item];

		std::string breakDownName = "breakDown//" + name + ".txt";
		FILE* breakDownFile;


		fopen_s(&breakDownFile, breakDownName.c_str(), "w+");



		std::cerr << "file: " << item + 1 << "//" << names.size() << std::endl;
		cudaDeviceReset();

		timer.update();

		//Step 0: ��ͼ, CPU�˺�ʱԼ600-800ms������󿽱�ԭʼ�����ݵ�GPU��ʱԼ40ms
		std::cerr << "Loading image.." << std::endl;
		string inputName = "fix-P7-4.5h-cell2-60x-zoom1.5_merge_c2.tif";

		inputName = file;


		//std::cerr << "Files: " << std::endl;
		//for (auto file : files)
		//{
		//	std::cerr << file << std::endl;
		//}

		//std::cerr << "Names: " << std::endl;
		//for (auto name : names)
		//{
		//	std::cerr << name << std::endl;
		//}

		/*if (argc > 1)
		{

			inputName = argv[1];

			if (inputName == "3")
				inputName = "case1-slide2-section2-left-cell3_merge_c2.tif";

			if (inputName == "5")
				inputName = "fix-P7-4.5h-cell2-60x-zoom1.5_merge_c2.tif";
		}

		inputName = "data//" + inputName;*/

		int *imageShape = new int[3];
		uchar* h_imagePtr = loadImage(inputName, imageShape);
		int width = imageShape[0]; //963
		int height = imageShape[1]; //305
		int slice = imageShape[2]; //140
		std::cerr << "Image Reading cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;

		timer.update();
		timer2.update();


		//Step 1: GPUԤ��

		warmupKernel << <1, 1 >> > ();

		printf("warmup cost: %dms\n\n", (int)(timer.getTimerMilliSec()));
		timer.update();
		timer3.update();
		uchar* d_imagePtr;
		cudaMalloc((void**)&d_imagePtr, sizeof(uchar) * width * height * slice);
		//��ԭͼ��host���Ƶ�device
		cudaMemcpy(d_imagePtr, h_imagePtr, sizeof(uchar) * width * height * slice, cudaMemcpyHostToDevice);
		std::cerr << "first malloc and memcpy cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();


		int g2 = getGlobalThreshold(h_imagePtr, d_imagePtr, width, height, slice);

		std::cerr << "get globalThreshold cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();



		//Step 2: �����ֵ�Ȼ���Ԥ����

		//����blockSize�ֿ飬���ݿ��ڻҶ��������Ӧ�����ֵ
		//��ʱ:Լ20ms
		int localThresholdBlockSize = 32;


		bool useLocalTh = true;

		//useLocalTh = false; // for flycircuit

		/*if (argc > 3)
		{

			string ifUseLocal = argv[3];

			if (ifUseLocal == "0")
				useLocalTh = false;
		}*/


		if (useLocalTh)
			addLocalThreshold(d_imagePtr, width, height, slice, localThresholdBlockSize);
		std::cerr << "add localThreshold cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();


		//���ȫ����ֵ
		//��ʱ��<1ms
		int globalThreshold = 5;
		globalThreshold = g2;

		if (name == "1_1_Live_2-2-2010_9-52-24_AM_med_Red.tif")
			globalThreshold = 50;
		if (name == "1_5dpf_Live_1-30-2010_12-39-26_PM_med_Red.tif")
			globalThreshold = 60;
		if (name == "lAPT_PN1_neuron.tif")
			globalThreshold = 30;
		if (name == "SLP_PN_neuron.tif")
			globalThreshold = 35;
		if (name == "WED_SLP_PN_neuron.tif")
			globalThreshold = 30;

		/*if (argc > 2)
		{
			if (atoi(argv[2]) != -1)
				globalThreshold = atoi(argv[2]);
		}*/

		//globalThreshold = 1;// for flycircuits

		if (globalThreshold > 1)
			addGlobalThreshold(d_imagePtr, width, height, slice, globalThreshold);
		std::cerr << "Add GlobalThreshold cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();

		//���㹻�������򣬽�����Χ�İ�����ֵΪ1���������֮��ķ�϶
		//��ʱ: <1ms
		int paddingThreshold = globalThreshold + 2;
		addDarkPadding(d_imagePtr, width, height, slice, paddingThreshold);
		cudaDeviceSynchronize();
		std::cerr << "Add Padding cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();


		//Step 3: ѹ��ԭʼͼ�񣬳�ȥ��0ֵ (��Ϊstream compaction, ��ѹ��) 
		//��ʱ:Լ70ms


		//������������洢��ԭͼ��ѹ����ͼ֮���ӳ��
		//compress: global Idx-> compressed Idx
		//decompress: compressed Idx-> global Idx
		int* d_compress;
		int* d_decompress;
		uchar* d_imagePtr_compact;

		//������ָ�붼�ں����������ռ�,��Ϊd_decompressδ֪��С
		int newSize = -1;

		compactImage(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, width, height, slice, newSize);
		std::cerr << "Old: " << width * height * slice << " New:" << newSize << std::endl;

		cudaDeviceSynchronize();
		std::cerr << "Stream Compaction cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();


		//while (newSize > width * height * slice / 20 && globalThreshold < 250)
		//{
		//	
		//	if (globalThreshold < 245)
		//		globalThreshold += 2;
		//	else
		//		globalThreshold = 250;
		//	std::cerr << "auto adapting globalth: " << globalThreshold << std::endl;
		//	

		//	addGlobalThreshold(d_imagePtr, width, height, slice, globalThreshold);
		//	std::cerr << "Add GlobalThreshold cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		//	timer.update();

		//	paddingThreshold = globalThreshold + 2;
		//	addDarkPadding(d_imagePtr, width, height, slice, paddingThreshold);
		//	cudaDeviceSynchronize();
		//	std::cerr << "Add Padding cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		//	timer.update();

		//	cudaFree(d_compress);
		//	cudaFree(d_decompress);
		//	cudaFree(d_imagePtr_compact);

		//	d_compress = NULL;
		//	d_decompress = NULL;
		//	d_imagePtr_compact = NULL;

		//	compactImage(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, width, height, slice, newSize);
		//	std::cerr << "Old: " << width * height * slice << " New:" << newSize << std::endl;

		//	cudaDeviceSynchronize();
		//	std::cerr << "Stream Compaction cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		//	timer.update();
		//}




		//Step 5: ����ÿ����Ŀ��ư뾶���������ļ�֦�ж�
		//��ʱ:Լ70ms
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


		//Step 4: GrayWeight Distacne Transform, �ҶȾ���任����ΪԤ�����һ�֣���ͼ�������
		//��ʱ:Լ80ms

		addGreyWeightTransform(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, width, height, slice, newSize);

		std::cerr << "GWDT cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();

		int maxpos = 0;


		////Ѱ�����ֵ����Ϊ��Ԫ��������
		//thrust::device_ptr<uchar> d_ptr(d_radiusMat);
		//thrust::device_ptr<uchar> iter = thrust::max_element(d_ptr, d_ptr + width * height * slice);
		//int maxRadius = (int)(*iter);
		//maxpos = iter - d_ptr;


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

		fprintf(breakDownFile, "%.2lf ", timer3.getTimerMilliSec());

		timer3.update();


		//Step 6: ���ɲ���,�������׷���ṩ���ӡ�ʵ����Wei et al.��parallel poisson disk sampling
		//��ʱ:Լ120ms
		//��Ϊ������ɸѡ�������裬ɸѡ��Ҫ����ԭ���飨�㹻���Ĳ�Ҫ�������д���������ĺ����İ뾶
		//���ص���std::vector seedArr������洢�����ӵ��λ��, ͬʱ���������������һ��Ԫ�أ��������ģ���Ϊ�����Ҳ������Ҫ������


		std::vector<int> seedArr;

#ifdef __ONLY__ONE__SEED
		seedArr.push_back(center.z * width * height + center.y * width + center.x);
#else
		doPoissonSample2(seedArr, center, maxRadius, width, height, slice, newSize, d_imagePtr, d_imagePtr_compact, d_compress, d_decompress);
		//doPoissonSample/doPoissonSample2 ���в����������������

#endif // __ONLY__ONE__SEED





		cudaDeviceSynchronize();
		std::cerr << "GPU sampling postprocessing and output cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();

		//doPoissonSample_cpu(seedArr, center, maxRadius, width, height, slice, newSize, d_imagePtr, d_imagePtr_compact, d_compress, d_decompress);

		//std::cerr << "GPU sampling postprocessing and output cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		//timer.update();

		//doPoissonSample2(seedArr, center, maxRadius, width, height, slice, newSize, d_imagePtr, d_imagePtr_compact, d_compress, d_decompress);

		//std::cerr << "GPU sampling postprocessing and output cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		//timer.update();

		cudaDeviceSynchronize();
		errorCheck = cudaGetLastError();
		if (errorCheck != cudaSuccess) {
			std::cerr << "During Poisson Sample " << cudaGetErrorString(errorCheck) << std::endl;
			system("pause");
			return;
		}

		std::cerr << "[Breakdown] Total Seed Generating Cost: " << timer3.getTimerMilliSec() << "ms" << std::endl << std::endl;
		fprintf(breakDownFile, "%.2lf ", timer3.getTimerMilliSec());
		timer3.update();

		//Step 7: ��ʼ׷��,ʹ�ü򻯵�fast marching�㷨������˵�������·�㷨
		//��ʱ:Լ250ms

		short int* d_seedNumberPtr;
		uchar* d_activeMat_compact;
		cudaMalloc((void**)&d_activeMat_compact, sizeof(uchar) * newSize);
		cudaMemset(d_activeMat_compact, FARAWAY, sizeof(uchar) * newSize);

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
		fprintf(breakDownFile, "%.2lf ", timer3.getTimerMilliSec());
		timer3.update();

		//Step 8: ��֧�ϲ��㷨
		//��ʱ:Լ50ms
		std::vector<int> leafArr;
		int leafCount = 0;
		int validLeafCount = 0;

		//���鼯�������жϲ�ͬ��֧֮���Ƿ�����˺ϲ�
		std::vector<int> disjointSet(seedArr.size() + 1, 0);
		for (int it = 0; it < disjointSet.size(); it++)
			disjointSet[it] = it;

		int* d_disjointSet;
		cudaMalloc(&d_disjointSet, sizeof(int) * disjointSet.size());
		cudaMemcpy(d_disjointSet, &(disjointSet[0]), sizeof(int) * disjointSet.size(), cudaMemcpyHostToDevice);




		//std::cerr << "disjointSet size: " << disjointSet.size() <<  std::endl;
		mergeSegments(seedArr, disjointSet, width, height, slice, newSize, d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_childNumMat, d_radiusMat_compact, d_parentPtr_compact, d_seedNumberPtr, d_disjointSet);

		std::cerr << "MergeSegments cost: " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
		timer.update();

		std::cerr << "[Breakdown] Total Merging Cost: " << timer3.getTimerMilliSec() << "ms" << std::endl << std::endl;
		fprintf(breakDownFile, "%.2lf ", timer3.getTimerMilliSec());
		timer3.update();

		cudaDeviceSynchronize();
		errorCheck = cudaGetLastError();
		if (errorCheck != cudaSuccess) {
			std::cerr << "Before Pruning " << cudaGetErrorString(errorCheck) << std::endl;
			system("pause");
			return;
		}


		///Step 9: ���ռ�֦
		//��ʱ:Լ350ms
		pruneLeaf_3d_gpu(leafArr, validLeafCount, disjointSet, width, height, slice, newSize, d_radiusMat_compact, d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_parentPtr_compact, d_activeMat_compact, d_childNumMat, d_seedNumberPtr, d_disjointSet);
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
		fprintf(breakDownFile, "%.2lf ", timer3.getTimerMilliSec());
		timer3.update();


		std::cerr << "Total time cost: " << timer2.getTimerMilliSec() << "ms" << std::endl;
		fprintf(breakDownFile, "%.2lf\n", timer2.getTimerMilliSec());
		timer2.update();

		
		fclose(breakDownFile);
	}

	return 0;
}



