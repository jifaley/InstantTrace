#include "threshold.h"

static __constant__ const int dx3dconst[6] = { -1, 1, 0, 0, 0, 0 };
static __constant__ const int dy3dconst[6] = { 0, 0, -1, 1, 0, 0 };
static __constant__ const int dz3dconst[6] = { 0, 0, 0, 0, -1, 1 };

static __constant__ const int dx3d26const[26] = { -1,-1,-1,-1,-1,-1,-1,-1,-1,   0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1, 1 };
static __constant__ const int dy3d26const[26] = { -1,-1,-1, 0, 0, 0, 1, 1, 1,  -1,-1,-1, 0, 0, 1, 1, 1, -1,-1,-1, 0, 0, 0, 1, 1, 1 };
static __constant__ const int dz3d26const[26] = { -1, 0, 1,-1, 0, 1,-1, 0, 1,  -1, 0, 1,-1, 1,-1, 0, 1, -1, 0, 1,-1, 0, 1,-1, 0, 1 };


struct is_less_than_th
{
	is_less_than_th(uchar th = 0) :_th(th){}
	__host__ __device__
		bool operator()(int x)
	{
		return x < _th;
	}
private:
	uchar _th;
};

struct getVar : thrust::unary_function<uchar, double>
{
	getVar(double mean) : _mean(mean){}
	const double _mean;
	__host__ __device__ double operator()(uchar data) const
	{
		return (data - _mean) * (data - _mean);
	}
};

using thrust::placeholders::_1;

int getGlobalThreshold(uchar* h_imagePtr, uchar* d_imagePtr, int width, int height, int slice)
{
	double sum = 0;
	//for (int i = 0; i < width * height * slice; i++)
	//{
	//	sum += h_imagePtr[i];
	//}

	sum = thrust::reduce(thrust::device, d_imagePtr, d_imagePtr + width * height * slice, 0.0, thrust::plus<double>());

	double mean = sum / (width * height * slice);

	double var = 0;


	var = thrust::transform_reduce(thrust::device, d_imagePtr, d_imagePtr + width * height * slice, getVar(mean), 0.0, thrust::plus<double>());
	

	var = var / (width * height * slice - 1);


	double std = sqrt(var);

	double td = (std < 10) ? 10 : std;

	printf("mean: %.2lf, std:%.2lf\n", mean, std);

	int th = mean + 0.5 * td;

	printf("autoset global Th = %d\n", th);

	return th;

}


/*
������addGlobalThreshold
���ܣ���d_imagePtr ָ���ͼ�����ȫ����ֵ
*/
void addGlobalThreshold(uchar* d_imagePtr, int width, int height, int slice, uchar threshold)
{
	is_less_than_th comp(threshold);
	thrust::replace_if(thrust::device, d_imagePtr, d_imagePtr + width * height * slice, d_imagePtr, comp, 0);
}

/*
������addLocalThreshold_Kernel
���ܣ���d_imagePtr ָ���ͼ����Ӿֲ���ֵ
ʵ�֣����ȸ���blockSize������ͼ��ֿ飬�ֱ�ͳ�ƻҶ�ֱ��ͼ��ֻ����������������ǰ5%��ֵ��
���ݣ���Ԫ�ź�һ����Ա�����˵�������ġ�
ȱ�㣺�������Ϊ���Եķֿ�ЧӦ��Ӧ����Ӳ�ֵ������������
*/
__global__
void addLocalThresholdKernel(uchar* inputPtr, int width, int height, int slice, int blockSize, int kmax, int imax, int jmax, int* d_localThresholdArr)
{
	//valueCount: �洢�Ҷ�ֱ��ͼ
	__shared__ int valueCount[256];
	//valueCountCumulate: �洢ֱ��ͼ��ǰ׺��
	volatile __shared__ int valueCountCumulate;
	//blockPixelCount: С���ڵ���������
	volatile __shared__ int blockPixelCount;
	//locakThreshold: С���ڵ���ֵ
	volatile __shared__ int localThreshold;
	volatile __shared__ int k_id, i_id, j_id, kStart, iStart, jStart;

	int bid = blockIdx.y * gridDim.x + blockIdx.x;

	if (bid >= kmax * imax * jmax) return;
	int tid = threadIdx.x;
		
	for (int i = tid; i < 256; i += blockDim.x)
	{
		valueCount[i] = 0;
	}
	__syncthreads();


	if (tid == 0)
	{
		valueCountCumulate = 0;
		k_id = bid / (imax * jmax);
		i_id = bid % (imax * jmax) / jmax;
		j_id = bid % jmax;

		kStart = k_id * blockSize;
		iStart = i_id * blockSize;
		jStart = j_id * blockSize;

		blockPixelCount = MIN(blockSize, slice - kStart) * MIN(blockSize, height - iStart) * MIN(blockSize, width - jStart);
	}

	__syncthreads();

	int temp, i, j, k;
	for (k = kStart; k < kStart + blockSize && k < slice; k++)
		for (i = iStart; i < iStart + blockSize && i < height; i++)
			for (j = jStart + tid; j < jStart + blockSize && j < width; j += blockDim.x)
			{
				temp = inputPtr[k * width * height + i * width + j];
				atomicAdd(valueCount + temp, 1);
			}

	__syncthreads();
	if (tid == 0)
	{

		valueCountCumulate = valueCount[0];
		for (int it = 1; it <= 255; it++)
		{
			valueCountCumulate += valueCount[it];
			if (valueCountCumulate <= blockPixelCount * 0.90)
			{
				localThreshold = it;
			}
			else
				break;
		}
		d_localThresholdArr[bid] = localThreshold;
		localThreshold = d_localThresholdArr[bid];
	}

	__syncthreads();

	for (k = kStart; k < kStart + blockSize && k < slice; k++)
		for (i = iStart; i < iStart + blockSize && i < height; i++)
			for (j = jStart + tid; j < jStart + blockSize && j < width; j += blockDim.x)
			{
				temp = inputPtr[k * width * height + i * width + j];

				//temp < 20��һ����������ɾ�������Ĳ���
				//if (temp <= localThreshold && temp < 20)
				//	inputPtr[k * width * height + i * width + j] = 0;

				if (temp <= localThreshold)
					inputPtr[k * width * height + i * width + j] = 0;
			}
}


/*
������addLocalThreshold
���ܣ���d_imagePtr ָ���ͼ����Ӿֲ���ֵ
ʵ�֣����ȸ���blockSize������ͼ��ֿ飬�ֱ�ͳ�ƻҶ�ֱ��ͼ��ֻ����������������ǰ5%��ֵ��
���ݣ���Ԫ�ź�һ����Ա�����˵�������ġ�
ȱ�㣺�������Ϊ���Եķֿ�ЧӦ��Ӧ����Ӳ�ֵ������������
*/
void addLocalThreshold(uchar* d_imagePtr, int width, int height, int slice, int blockSize)
{
	int kmax = (slice - 1) / blockSize + 1;
	int imax = (height - 1) / blockSize + 1;
	int jmax = (width - 1) / blockSize + 1;

	int totalBlock = kmax * imax * jmax;
	//����ÿ��block�ľֲ���ֵ
	int* d_localThresholdArr;
	
	cudaMalloc(&d_localThresholdArr, sizeof(int) * totalBlock);
	cudaMemset(d_localThresholdArr, 0, sizeof(int) * totalBlock);

	cudaError_t errorCheck;
	//cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "Before Localth " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	std::cerr << "TotalBlock:"<<  totalBlock << std::endl;
	addLocalThresholdKernel << < totalBlock, 32 >> > (d_imagePtr, width, height, slice, blockSize, kmax, imax, jmax,  d_localThresholdArr);


	//cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "During Localth " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	cudaFree(d_localThresholdArr);
}

/*
������addDarkPaddingKernel
���ܣ���d_imagePtr ָ���ͼ����в���
ʵ�֣������㹻�������򣬽����ܱߵİ����Ҷ���Ϊ1
���ݣ���ͼ���ͬ����֮��ķ�϶��ʹ�ú���׷��ʱ�ܳɹ��������ڵ�����
*/
__global__
void addDarkPaddingKernel(uchar* d_imagePtr, int width, int height, int slice, uchar threshold)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= width * height * slice) return;

	uchar curValue = d_imagePtr[idx];
	if (curValue >= threshold)
	{
		int3 curPos;
		curPos.z = idx / (width * height);
		curPos.y = idx % (width * height) / width;
		curPos.x = idx % width;
		//printf("%d %d %d\n", curPos.x, curPos.y, curPos.z);

		int3 neighborPos;
		int neighborIdx;
		for (int k = 0; k < 26; k++)
		{
			neighborPos.x = curPos.x + dx3d26const[k];
			neighborPos.y = curPos.y + dy3d26const[k];
			neighborPos.z = curPos.z + dz3d26const[k];
			if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
				|| neighborPos.z < 0 || neighborPos.z >= slice)
				continue;
			neighborIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;
			if (d_imagePtr[neighborIdx] == 0)
			{
				d_imagePtr[neighborIdx] = 1;
			}
		}

		int windowSize = 1;

		for (int dx = -windowSize; dx <= windowSize; dx++)
			for (int dy = -windowSize; dy <= windowSize; dy++)
				for (int dz = -windowSize; dz <= windowSize; dz++)
				{
					neighborPos.x = curPos.x + dx;
					neighborPos.y = curPos.y + dy;
					neighborPos.z = curPos.z + dz;
					if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
						|| neighborPos.z < 0 || neighborPos.z >= slice)
						continue;
					neighborIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;
					if (d_imagePtr[neighborIdx] == 0)
					{
						d_imagePtr[neighborIdx] = 1;
					}
				}
	}
}

/*
������addDarkPadding
���ܣ���d_imagePtr ָ���ͼ����в���
ʵ�֣������㹻�������򣬽����ܱߵİ����Ҷ���Ϊ1
���ݣ���ͼ���ͬ����֮��ķ�϶��ʹ�ú���׷��ʱ�ܳɹ��������ڵ�����
*/

void addDarkPadding(uchar* d_imagePtr, int width, int height, int slice, uchar threshold)
{
	addDarkPaddingKernel << <(width * height * slice - 1) / 256 + 1, 256 >> > (d_imagePtr, width, height, slice, threshold);
}


