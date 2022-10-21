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

//getCompressMap:compactImage���Ӻ��������ڼ�����ѹ�����ӳ�䡣d_compressΪԪ���±�->ѹ���±꣬d_decompress��֮��
//getCompressMap:The sub-function of compactImage. Calculating the mapping for stream compaction. The "d_compress" array
//is the mapping from the original element index to the compressed element index. The "d_decompress" array is the inversed mapping.
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
/*
Function��compactImage
Work��Compress the original image, leave out the zero-valued elements. (Also known as Stream Compaction)
Output��d_compactedImagePtr(The compressed image)��d_compress (The compression mapping)��d_decompress(The decompression mapping)
Implementaion��Binding the voxels and their indices to tuples, as the form of (0,value0), (1, value1), (2,value2)....
After deleting the zero-valued tuples, the remainders are arranged as (id0, value_id0), (id1, value_id1)...
Thus, these values form the compressed image, and these ids are the corresponding indices in the orginal image.
This function can be implemented by thrust::copy_if or thrust::remove_if.
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


//���ܣ�����d_sequence������Ԫ�ص�x,y,z����ƽ��ֵ��Ȼ���ҵ�����ƽ��ֵ�����Ԫ�ء�
//Work��Calculating the average of x,y,z coordinates in the d_sequence array��and find the element nearest to this average coordinate.
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

/*
������getCenterPos
���ܣ�Ѱ��Radius���ĵ㣬��Ϊ����(soma)
�����maxPos(�����λ��)��maxRadius(���뾶)
˼·�����ֻ�Ұ뾶���ĵ㣬���ܻ��кܶ���ͬ��ȡֵ������ƫб��
��ˣ����ǽ���Χ�뾶�㹻������ɵ��λ�ü���ƽ��ֵ����Ϊ�µİ������ġ�
ʵ�֣�ʹ��thrust���copy_if ���� remove_if ����
*/
/*
Function��getCenterPos
Work��Find the point with the largest radius, as the center of neuron soma.
Output��maxPos(the location of soma)��maxRadius(the largest radius)
Implementation��The element with the largest radius may not locates at the neuron center.
We generate a lot of candidates with large radius, and calculate the center of them as the neuron center.
*/
void getCenterPos(int* d_compress, int* d_decompress, uchar* d_radiusMat_compact, int width, int height, int slice, int newSize, int&maxPos, int& maxRadius)
{
	thrust::device_ptr<uchar> d_ptr(d_radiusMat_compact);
	thrust::device_ptr<uchar> iter = thrust::max_element(d_ptr, d_ptr + newSize);
	maxRadius = *iter;
	//����ͨ��max_element��������뾶��ֵ
	//Find the largest radius

	printf("Init maxRadius: %d\n", maxRadius);
	
	int* d_sequence;
	cudaMalloc(&d_sequence, sizeof(int) * newSize);

	//���ǽ����뾶��4/5�������뾶-5��Ϊ��ֵ��ѡ��һЩ��ѡ�㣻����Щ��ѡ���������Ϊ�������ġ�
	//The threshold radius for generating center candidates
	uchar thresholdRadius = MAX(maxRadius * 4 / 5, maxRadius - 5);

	int* d_copy_end = thrust::copy_if(thrust::device, thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(newSize), d_radiusMat_compact, d_sequence,isValid_functor(thresholdRadius));
	int maxSeedNum = d_copy_end - d_sequence;

	maxSeedNum = MIN(maxSeedNum, 512);

	//����d_sequence������Ԫ�ص�x,y,z����ƽ��ֵ��Ȼ���ҵ�����ƽ��ֵ�����Ԫ�ء�
	centerProcess << <1, maxSeedNum >> > (d_sequence, d_decompress, maxSeedNum, width, height, slice);

	thrust::device_ptr<int> dp(d_sequence);

	maxPos = *dp;
	cudaFree(d_sequence);
}