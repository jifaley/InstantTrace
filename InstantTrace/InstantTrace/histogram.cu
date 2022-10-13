#include "histogram.h"



__global__
void setZero(int* valueCount, int* valueCountCumulate, int* valueMap)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= 256) return;
	valueCount[idx] = 0;
	valueCountCumulate[idx] = 0;
	valueMap[idx] = 0;
}

__global__
void valueCountCal(int *valueCount, uchar *outputPtr,
	int iStart, int jStart, int kStart, int blockSize,
	int height, int width, int slice)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + iStart;//i 加上 iStart 来定位
	int j = threadIdx.y + blockIdx.y * blockDim.y + jStart;
	int k = threadIdx.z + blockIdx.z * blockDim.z + kStart;
	if ((i) >= height || (i - iStart) >= blockSize) return;
	if ((j) >= width || (j - jStart) >= blockSize) return;
	if ((k) >= slice || (k - kStart) >= blockSize) return;

	atomicAdd(&(valueCount[(int)(outputPtr[k * width * height + i * width + j])]), 1);
}

__global__
void valueMapCal(int *valueMap, int blockPixelCount, int *valueCountCumulate)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	valueMap[idx] = (int)((valueCountCumulate[idx] - valueCountCumulate[0]) * 1.0 / (blockPixelCount - valueCountCumulate[0]) * 255);
}

__global__
void blockPixelCountCal(int *valueCountCumulate, int* valueMap, int blockPixelCount)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (valueCountCumulate[idx] < blockPixelCount* 0.99) //值太小的 直接归零
		valueMap[idx] = 1;
}

__global__
void outputCal(int *valueMap, uchar *outputPtr, uchar *d_block,
	int iStart, int jStart, int kStart, int blockSize,
	int height, int width, int slice)
{
	int j = threadIdx.x + blockIdx.x * blockDim.x + jStart;//i 加上 iStart 来定位
	int i = threadIdx.y + blockIdx.y * blockDim.y + iStart;
	int k = threadIdx.z + blockIdx.z * blockDim.z + kStart;
	if (i >= height || (i - iStart) >= blockSize) return;
	if (j >= width || (j - jStart) >= blockSize) return;
	if (k >= slice || (k - kStart) >= blockSize) return;

	uchar temp = outputPtr[k * width * height + i * width + j];
	outputPtr[k * width * height + i * width + j] = valueMap[temp];
}

__global__
void valueCountToCumulateAndvalueMapToZero(int* valueCountCumulate, int* valueCount, int* valueMap)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx == 0)
	{
		valueCountCumulate[idx] = valueCount[idx];
		valueMap[idx] = 0;
	}
}

__global__
void MemsetTo1(int *d_value)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	d_value[idx] = 1;
}

__global__
void blockCal(int *valueMap, uchar *outputPtr, uchar *d_block,//搞到cpu上
	int iStart, int jStart, int kStart, int blockSize,
	int height, int width, int slice)
{
	int j = threadIdx.x + blockIdx.x * blockDim.x + jStart;//i 加上 iStart 来定位
	int i = threadIdx.y + blockIdx.y * blockDim.y + iStart;
	int k = threadIdx.z + blockIdx.z * blockDim.z + kStart;
	if (i >= height || (i - iStart) >= blockSize) return;
	if (j >= width || (j - jStart) >= blockSize) return;
	if (k >= slice || (k - kStart) >= blockSize) return;

	uchar temp = outputPtr[k * width * height + i * width + j];

	int blockHeight = MIN(blockSize, height - iStart);
	int blockWidth = MIN(blockSize, width - jStart);

	d_block[(j - jStart) + (i - iStart) * blockWidth + (k - kStart) * blockHeight * blockWidth] = temp;//把output里的一块图像集中到连续的一条内存 block 里面， 再把它与之后的output替换 这样就可以得到需要的图像了
}

__global__
void histogramComplete(uchar *d_C, int *d_D, int *valueCountCumulate)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= 256) return;
	if (!d_D[idx]) return;
	int index = d_C[idx];
	valueCountCumulate[index] = d_D[idx];
}

void equalizeHistogramGpu(uchar* const inputPtr, uchar* const outputPtr, int width, int height, int slice, int *sizes, uchar* d_output)
{

	int *valueCount, *valueCountCumulate, *valueMap;
	int blockSize = 256;
	int blockPixelCount = 0;
	//const int blockNum = (width / blockSize + 1) * (height / blockSize + 1) * (slice / blockSize + 1);
	//Modified by jifaley 20210830
	const int blockNum = ((width - 1) / blockSize + 1) * ((height - 1) / blockSize + 1) * ((slice - 1) / blockSize + 1);
	std::cout << "blockNum: " << blockNum << std::endl;
	cudaMalloc((void**)&valueCount, sizeof(int) * 256 * blockNum);
	cudaMalloc((void**)&valueCountCumulate, sizeof(int) * 256 * blockNum);
	cudaMalloc((void**)&valueMap, sizeof(int) * 256 * blockNum);

	//cudaMallocHost((void**)&d_output, sizeof(uchar)*(width*height*slice));//pine

	//cudaMemset(d_value, 0,sizeof(int)*(width*height*slice));

	cudaMemcpy(d_output, inputPtr, sizeof(uchar)*(width*height*slice), cudaMemcpyHostToDevice);
	//std::string save_dir_f_ = "image(single)/neuron01_kernel_before.tif";
	//saveTiff(save_dir_f_.c_str(), inputPtr, &sizes[0]);

	//d_block:处理单块 d_C d_D 用来 reduece_by_key
	uchar *d_block;
	int *d_value;
	uchar *d_C;
	int *d_D;

	cudaMalloc((void**)&d_block, sizeof(uchar)*(width*height*slice));
	cudaMalloc((void**)&d_value, sizeof(int)*blockSize*blockSize*blockSize);
	cudaMalloc((void**)&d_C, sizeof(uchar)*blockSize*blockSize*blockSize);
	cudaMalloc((void**)&d_D, sizeof(int)*blockSize*blockSize*blockSize);
	//set some streams to overlap the kernels or the memcpy
	//假设只处理512 512 512 图片， 并且已固定尺寸 128
	//const int MAX_NUM_STREAMS = 1480;
	//cudaStream_t streams[MAX_NUM_STREAMS];
	//for (int n = 0; n < MAX_NUM_STREAMS; ++n)
	//{
	//	cudaStreamCreate(&(streams[n]));
	//}
	//int STREAMS = 0;
	//设置d_block的索引 offset

	//Modified by jifaley 20210830
	int offset = 0;
	//value_offset
	int value_offset = 0;

	//改成多个 for loop 并指定stream
//#pragma unroll

	thrust::device_ptr<uchar> d_block_thrust(d_block);
	thrust::device_ptr<int> d_value_thrust(d_value);

	for (int kStart = 0; kStart < slice; kStart += blockSize)
		for (int iStart = 0; iStart < height; iStart += blockSize)
			for (int jStart = 0; jStart < width; jStart += blockSize) {
				//offset += blockSize * blockSize*blockSize;


				//01
				dim3 block_size(256, 1, 1);
				dim3 grid_size(1, 1, 1);
				setZero << <grid_size, block_size >> > (valueCount + value_offset, valueCountCumulate + value_offset, valueMap + value_offset);

				//02
				dim3 grid_size_blockCal(1, MIN(blockSize, height - iStart), MIN(blockSize, width - jStart));
				//改
				dim3 block_size_blockCal(MIN(blockSize, slice - kStart), 1, 1);

				blockCal << <grid_size_blockCal, block_size_blockCal >> > (valueMap + value_offset, d_output, d_block + offset,
					iStart, jStart, kStart, blockSize,
					height, width, slice);

				//03
				blockPixelCount = MIN(blockSize, slice - kStart) * MIN(blockSize, height - iStart) * MIN(blockSize, width - jStart);

				dim3 block_size_1(256, 1, 1);
				dim3 grid_size_1((blockPixelCount + 256 - 1) / 256, 1, 1);

				//计算直方图时用到 sort_by_key 其中 key 全为一
				MemsetTo1 << <grid_size_1, block_size_1 >> > (d_value);

				//04
				blockPixelCount = MIN(blockSize, slice - kStart) * MIN(blockSize, height - iStart) * MIN(blockSize, width - jStart);
				thrust::sort(thrust::device, d_block + offset, d_block + offset + blockPixelCount);


				//05
				blockPixelCount = MIN(blockSize, slice - kStart) * MIN(blockSize, height - iStart) * MIN(blockSize, width - jStart);

				thrust::equal_to<uchar> binary_pred;
				thrust::reduce_by_key(thrust::device, d_block + offset, d_block + offset + blockPixelCount, d_value, d_C, d_D, binary_pred);

				//把d_C d_D 映射成完整的样子
				//uchar *d_blockHistogram;  原来打算新开一个  d_blockHistogram  后面直接用了老的 valueCountCumulate
				//cudaMalloc((void**)&d_blockHistogram, sizeof(uchar) * 256);
				dim3 blockComplete(256, 1, 1);
				dim3 gridComplete(1, 1, 1);
				histogramComplete << <gridComplete, blockComplete >> > (d_C, d_D, valueCountCumulate + value_offset);

				thrust::inclusive_scan(thrust::device, valueCountCumulate + value_offset, valueCountCumulate + value_offset + 256, valueCountCumulate + value_offset);

				//06
				dim3 block_size_3(256, 1, 1);
				dim3 grid_size_3(1, 1, 1);
				valueMapCal << <grid_size_3, block_size_3 >> > (valueMap + value_offset, blockPixelCount, valueCountCumulate + value_offset);

				//07
				dim3 block_size_4(256, 1, 1);
				dim3 grid_size_4(1, 1, 1);
				blockPixelCountCal << <grid_size_4, block_size_4 >> > (valueCountCumulate + value_offset, valueMap + value_offset, blockPixelCount);


				//08
				dim3 grid_size_5(blockSize / 16, blockSize / 16, blockSize / 4);
				dim3 block_size_5(16, 16, 4);

				outputCal << <grid_size_5, block_size_5 >> > (valueMap + value_offset, d_output, d_block + offset,
					iStart, jStart, kStart, blockSize,
					height, width, slice);

				//Modified by jifaley 20210830
				offset += MIN(blockSize, height - iStart) * MIN(blockSize, width - jStart)*MIN(blockSize, slice - kStart);
				value_offset += 256;
			}


	cudaFree(d_block);
	cudaFree(d_value);
	cudaFree(d_C);
	cudaFree(d_D);
	cudaMemcpy(outputPtr, d_output, sizeof(uchar)*(width*height*slice), cudaMemcpyDeviceToHost);
	//int outputPtrInt = (int)outputPtr[6558462];
	//printf("outputPtr[6558462] = %d\n", outputPtrInt);
	//std::string save_dir_f = "image(single)/neuron01_kernel_after.tif";
	//saveTiff(save_dir_f.c_str(), outputPtr, &sizes[0]);

	cudaFree(valueCount);
	cudaFree(valueCountCumulate);
	cudaFree(valueMap);
	//Modified by jifaley 20210830
	//free(inputPtr); 不能释放，后面还要用
}
