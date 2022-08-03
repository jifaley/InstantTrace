#include "compaction.h"
#include "TimerClock.hpp"

template<typename T>
struct is_non_zero {
	__host__ __device__
		bool operator()(T x) const
	{
		return x != 0;
	}
};

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
void getCompressMap(int* d_compress, int* d_decompress, uchar* d_imagePtr, uchar* d_imagePtr_compact, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;
	int fullIdx = d_decompress[smallIdx];

	d_compress[fullIdx] = smallIdx;
	d_imagePtr_compact[smallIdx] = d_imagePtr[fullIdx];
}


/*
������compactImage
���ܣ�ѹ��ԭͼ��ȥ����0���֡� 
�����d_compactedImagePtr(ѹ�����ͼ)��d_compress (ԭͼ->ѹ��ͼӳ��)��d_decompress(ѹ��ͼ->ԭͼӳ�䣩
˼·�����Ƚ��������غ����±��Ϊtuple��������(0,value0), (1, value1), (2,value2)....
������value< 0�Ĳ���ɾ����ʣ���tuple��Ϊ: (id0, value_id0), (id1, value_id1)...
��ô,ʣ���valueֵ��Ϊѹ�����ͼ��ʣ���id��Ϊѹ�����ֵ��Ӧ��ԭͼ�е��±ꡣ
ʵ�֣�ʹ��thrust���copy_if ���� remove_if ����
*/

void compactImage(uchar* d_imagePtr, uchar* &d_imagePtr_compact, int* &d_compress, int* &d_decompress, int width, int height, int slice, int& newSize)
{
	TimerClock timer;
	timer.update();

	cudaError_t errorCheck;
	cudaMalloc(&d_compress, sizeof(int) * width * height * slice);
	int* d_sequence = d_compress; //ԭ�����������顣Ϊ�˽�ʡ��������ʱ����ͬһ��ռ�

	//������50ms���ҵ�ͬ��ʱ�䣨��ʹȥ��cuDeiveSyncronize()��cudaMemset()Ҳ��ǿ��ͬ����
	cudaDeviceSynchronize();
	std::cerr << "stream compaction preprocess cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();


	//����copy_if��d_sequence�����µ���ԭʼ�����ݷ�0ֵ���±ꡣ�ò�����stable�ġ� newSize��Ϊ��0ֵ�ĸ�����
	try
	{
		int* d_copy_end = thrust::copy_if(thrust::device, thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(width * height * slice), d_imagePtr, d_sequence, _1 != 0);
		newSize = d_copy_end - d_sequence;
	}
	catch (thrust::system_error error)
	{
		std::cerr << std::string(error.what()) << std::endl;
	}

	cudaMalloc(&d_decompress, sizeof(int) * newSize);
	cudaMalloc(&d_imagePtr_compact, sizeof(uchar) * newSize);
	cudaMemcpy(d_decompress, d_sequence, sizeof(int) * newSize, cudaMemcpyDeviceToDevice);
	cudaMemset(d_compress, 0xff, sizeof(int) * width * height * slice);

	//�����Ӧ��ӳ��
	getCompressMap << < (newSize - 1) / 256 + 1, 256 >> > (d_compress, d_decompress, d_imagePtr, d_imagePtr_compact, newSize);

	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "Duing copyif " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}
	//�������㣬����copy_if ��getMap()��ʵ�ʺ�ʱԼ20ms����������50ms��ͬ���������ۡ�
}

struct isValid_functor {

	const uchar threshold;

	isValid_functor(uchar _th) : threshold(_th) {}
	__host__ __device__
		bool operator()(const uchar& x) const
	{
		return x >= threshold;
	}
};


__global__
void centerProcess(int* d_sequence, int* d_decompress, int maxSeedNum, int width, int height, int slice)
{
	__shared__ int sumX, sumY, sumZ;
	__shared__ int minDist;
	__shared__ int minPos;


	int tid = threadIdx.x;
	if (tid >= maxSeedNum) return;

	int smallIdx = d_sequence[tid];
	int fullIdx = d_decompress[smallIdx];

	int z = fullIdx / (width * height);
	int y = fullIdx % (width * height) / width;
	int x = fullIdx % width;

	atomicAdd(&sumZ, z);
	atomicAdd(&sumY, y);
	atomicAdd(&sumX, x);

	__syncthreads();

	if (tid == 0)
	{
		sumX = sumX / maxSeedNum;
		sumY = sumY / maxSeedNum;
		sumZ = sumZ / maxSeedNum;
		minDist = 2147483647;
		d_sequence[0] = 2147483647;
	}

	__syncthreads();

	int dist = sqrtf((sumZ - z) * (sumZ - z) + (sumY - y) * (sumY - y) + (sumX - x) * (sumX - x));

	atomicMin(&minDist, dist);

	__syncthreads();

	if (minDist == dist)
	{
		atomicMin(&d_sequence[0], fullIdx);
	}
}

void getCenterPos(int* d_compress, int* d_decompress, uchar* d_radiusMat_compact, int width, int height, int slice, int newSize, int&maxPos, int& maxRadius)
{
	thrust::device_ptr<uchar> d_ptr(d_radiusMat_compact);
	thrust::device_ptr<uchar> iter = thrust::max_element(d_ptr, d_ptr + newSize);
	maxRadius = *iter;

	printf("Init maxRadius: %d\n", maxRadius);
	
	int* d_sequence;
	cudaMalloc(&d_sequence, sizeof(int) * newSize);

	uchar thresholdRadius = MAX(maxRadius * 4 / 5, maxRadius - 5);

	int* d_copy_end = thrust::copy_if(thrust::device, thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(newSize), d_radiusMat_compact, d_sequence,isValid_functor(thresholdRadius));
	int maxSeedNum = d_copy_end - d_sequence;

	maxSeedNum = MIN(maxSeedNum, 512);

	centerProcess << <1, maxSeedNum >> > (d_sequence, d_decompress, maxSeedNum, width, height, slice);

	thrust::device_ptr<int> dp(d_sequence);

	maxPos = *dp;
	cudaFree(d_sequence);
}