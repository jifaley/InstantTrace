#include "poissonSample.h"
#include "TimerClock.hpp"

#pragma comment(lib, "curand.lib")
static int max_level = 7;
static float d = 0.05f;


struct offset
{
	offset(int _x, int _y, int _z) {
		x = _x;
		y = _y;
		z = _z;
		dist = _x * _x + _y * _y + _z * _z;
	}
	int x;
	int y;
	int z;
	int dist;
};
bool compare_offset(const offset& offset1, const offset& offset2)
{
	return offset1.dist < offset2.dist;
}

int PGCompute(int x, int y, int z)
{
	x = x % 3 ? x % 3 : 3;
	y = y % 3 ? y % 3 : 3;
	z = z % 3 ? z % 3 : 3;

	return x + (y - 1) * 3 + (z - 1) * 3 * 3;
}

//cpu random generator
inline double UniformRandom()
{
	return static_cast<double>(rand()) / RAND_MAX;
}

//gpu random generator
struct psrngen
{
	__host__ __device__ psrngen(float _a, float _b) : a(_a), b(_b) { ; }

	__host__ __device__ float operator()(const unsigned int n) const
	{
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float> dist(a, b);
		rng.discard(n);
		return dist(rng);
	}
	float a, b;

};

//filterPoissonSample���Ӻ��������ڽ�����������Ĺ���/���ȹ��͵�sample����
//a sub-function of filterPoissonSample. Leave out the samples close to the neuron soma/the samples with low intensity.
__global__ 
void filterPoissonSampleKernel(int sampleNum, dim3 center, int centerRadius, 
	int poissonSampleThreshold, int width, int height, int slice, uchar* d_imagePtr, uchar* d_isSampleValid, int* d_seedPos, float* d_X_new_out, float* d_Y_new_out, float* d_Z_new_out)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= sampleNum) return;
	float x, y, z;
	int xInt, yInt, zInt;
	x = d_X_new_out[idx];
	if (x == 0) return;
	y = d_Y_new_out[idx];
	z = d_Z_new_out[idx];

	xInt = x * (width - 1) + 0.5f;
	yInt = y * (height - 1) + 0.5f;
	zInt = z * (slice - 1) + 0.5f;

	if ((xInt - center.x) * (xInt - center.x) + (yInt - center.y) * (yInt - center.y) + (zInt - center.z) * (zInt - center.z)
		< centerRadius * centerRadius * 16) return;

	if (d_imagePtr[zInt * width * height + yInt * width + xInt] > poissonSampleThreshold)
	{
		d_isSampleValid[idx] = 1;
		d_seedPos[idx] = zInt * width * height + yInt * width + xInt;
	}
}

/*
������filterPoissonSample_gpu
���ܣ�������������Ĺ���/���ȹ��͵�sample����
���룺d_X/Y/Z�� ��gpu���������ӵ��������飻center����������λ�ã�centerRadius������뾶
�����seedArr(���ɸѡ������ӵ��±�)
*/
/*
Function��filterPoissonSample_gpu
Work��Filter the random generated samples.The samples close to the center of neuron soma/the samples of low intensity are dropped.
Input��d_X/Y/Z�� The coordinate of the seeds; center��the location of neuron center; centerRadius��the control radius of neuron center
Output��seedArr (the indices of filtered seeds in the original image)
*/

void filterPoissonSample_gpu(std::vector<int>& seedArr, dim3 center, int centerRadius, int sampleNum, 
	int width, int height, int slice, uchar* d_imagePtr, float* d_X_new_out, float* d_Y_new_out, float* d_Z_new_out)
{
	//����Сkernel�Ƿ���cudamallocmanaged()��һ�㣿
	int validCount = 0;

	uchar* isSampleValid = (uchar*)malloc(sizeof(uchar) * sampleNum);
	memset(isSampleValid, 0, sizeof(uchar) * sampleNum);

	uchar* d_isSampleValid;
	cudaMalloc(&d_isSampleValid, sizeof(uchar) * sampleNum);
	cudaMemset(d_isSampleValid, 0, sizeof(uchar) * sampleNum);
	uchar poissonSampleThreshold = 10;
	poissonSampleThreshold = 1;

	int* seedPos = (int*)malloc(sizeof(int) * sampleNum);
	int* d_seedPos;
	cudaMalloc(&d_seedPos, sizeof(int) * sampleNum);



	//������������Ĺ���/���ȹ��͵�sample����
	//filter the samples
	filterPoissonSampleKernel << < (sampleNum - 1) / 256 + 1, 256 >> > (sampleNum, center, centerRadius, poissonSampleThreshold,
		width, height, slice, d_imagePtr, d_isSampleValid, d_seedPos, d_X_new_out, d_Y_new_out, d_Z_new_out);
	
	//����ʣ�����Ч���Ӹ���
	//calculating the number of valid samples
	validCount = thrust::reduce(thrust::device, d_isSampleValid, d_isSampleValid + sampleNum, 0, thrust::plus<int>());

	//ʣ���������̫���ˣ��ʵ��������������ֵ(Ҳ����ɾ���˽׶�)
	//if the number of valid samples are too much, increase the threshold. (This process can be removed.)
	while (validCount > 500 && poissonSampleThreshold <= 250)
	{
		validCount = 0;
		cudaMemset(d_isSampleValid, 0, sizeof(uchar) * sampleNum);
		poissonSampleThreshold += 5;

		filterPoissonSampleKernel << < (sampleNum - 1) / 256 + 1, 256 >> > (sampleNum, center, centerRadius, poissonSampleThreshold,
			width, height, slice, d_imagePtr, d_isSampleValid, d_seedPos, d_X_new_out, d_Y_new_out, d_Z_new_out);
		validCount = thrust::reduce(thrust::device, d_isSampleValid, d_isSampleValid + sampleNum, 0, thrust::plus<int>());

		printf("poissonSampleThreshold: %d, validCount: %d\n", poissonSampleThreshold, validCount);
	}

	cudaMemcpy(isSampleValid, d_isSampleValid, sizeof(uchar) * sampleNum, cudaMemcpyDeviceToHost);
	cudaMemcpy(seedPos, d_seedPos, sizeof(int) * sampleNum, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < sampleNum; i++)
	{
		if (isSampleValid[i])
		{
			seedArr.push_back(seedPos[i]);
		}
	}


	//������������Ϊ���һ������
	//Push the neuron center as the last seed point
	seedArr.push_back(center.z * width * height + center.y * width + center.x);
	std::cerr << "center idx: " << center.z * width * height + center.y * width + center.x << std::endl;
	std::cerr << "center_x: " << center.x << " center_y: " << center.y << " center_z: " << center.z << std::endl;
	validCount++;

	std::cerr << "Total num of samples GPU:" << sampleNum << std::endl;
	std::cerr << "Total num of valid samples GPU:" << validCount << std::endl;

	cudaFree(d_isSampleValid);
	cudaFree(d_seedPos);
	free(seedPos);
	free(isSampleValid);
}



//�������㲿��
//Poisson sampling kernel
__global__
void samplingKernel2(float* d_X_new, float* d_Y_new, float* d_Z_new,
	float* d_X_new_out, float* d_Y_new_out, float* d_Z_new_out,
	float* d_Random,
	int curSize, float r,
	int PG, int level,
	int* d_offset_x, int* d_offset_y, int* d_offset_z,
	uchar* d_imagePtr, uchar* d_visited, int* d_compress, int width, int height, int slice, int seedValueThreshold)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx >= curSize) return; //curSize�����ǵ�ǰPG�ĳ��ȱ��� level=2��PG=1, curSize = 4
	//���threadidx,���ֵֻ���ǵ�ǰPhase�ĸ�����������磺level=2, PG=1, ����Ϊ4
	//d_X_new�������4����㣬��ôthreadIdx����1-4��ÿ��ȡһ�����
	//printf("\n\nidx:%d\n", idx);
	//printf("cursize:%d\n", curSize);

	float curX = d_X_new[idx];
	float curY = d_Y_new[idx];
	float curZ = d_Z_new[idx];
	//printf("%d %f %f %f\n", idx, curX, curY, curZ);
	//printf("%f %f\n", curX, curY);
	//curX,curY�ǵ�ǰ�������ꡣͨ�����������Լ��㵱ǰ����index
	//���˶�ά�ĵ�ǰ���index,����offset���Ϳ��Լ���neighbor���Ķ�άindex.
	//ͨ��neighbor����һάindex�����Եõ�neighbor������sample�����ꡣ����d_X_new_out[neighhborIndex1d]
	//��ά���������(0,0),(3,0),(0,3),(3,3), һά���������0, 3, 12, 15

	//���񳤶�r
	int3 curIndex3d = { -1,-1,-1 };
	for (float t = 0; t < curX; t += r, curIndex3d.x += 1);
	for (float t = 0; t < curY; t += r, curIndex3d.y += 1);
	for (float t = 0; t < curZ; t += r, curIndex3d.z += 1);
	int ny = powf(2, level) + 0.5; //nx,ny,nz:ÿ��ά���ϵ��������
	int nx = ny;
	int nz = nx;
	int curIndex1d = curIndex3d.z * ny * nx + curIndex3d.y * nx + curIndex3d.x;
	//printf("%f %f %d %d %d\n", curX, curY, curIndex2d.x, curIndex2d.y, curIndex1d);
	//printf("d_X_new_out[curIndex1d] : %f\n", d_X_new_out[curIndex1d]);

	float r_x = width * 1.f / nx;
	float r_y = height * 1.f / ny;
	float r_z = slice * 1.f / nz; //ע��, ����<1�������㹻�����Ժ�> 1

	//printf("rx: %.2f, ry: %.2f, rz: %.2f\n", r_x, r_y, r_z);

	//�� ��PG�е����
	{
		if (d_X_new_out[curIndex1d] < 1e-5) //���������ǿյ�(=0)
		{
			//d_random()��[0,1]ȡֵ��-0.5������[-0.5,0.5]
			/*d_X_new_out[curIndex1d] = d_X_new[idx] + (d_Random[curIndex1d] - 0.5)*r;
			d_Y_new_out[curIndex1d] = d_Y_new[idx] + (d_Random[curIndex1d + 1] - 0.5)*r;
			d_Z_new_out[curIndex1d] = d_Z_new[idx] + (d_Random[curIndex1d + 2] - 0.5)*r;*/
			//r:��ǰ���񳤶ȡ�ʵ���������Ͷ��Ҳ���������������ڲ�Ͷ�㡣
			//��Ϊd_X_new[idx]���������ĵ����꣬�����ĳ���x,y,zƫ��0.5r��û�г�����

			bool find = false;

			int zstart = MAX(0, curIndex3d.z * r_z);
			int ystart = MAX(0, curIndex3d.y * r_y);
			int xstart = MAX(0, curIndex3d.x * r_x);

			int zend = MIN(slice, (curIndex3d.z + 1) * r_z);
			int yend = MIN(height, (curIndex3d.y + 1) * r_y);
			int xend = MIN(width, (curIndex3d.x + 1) * r_x);

			int tz, ty, tx;
			for (tz = zstart; tz < zend && find == false; tz++)
				for (ty = ystart; ty < yend && find == false; ty++)
					for (tx = xstart; tx < xend && find == false; tx++)
					{
						int fullIdx = tz * width * height + ty * width + tx;
						if (d_imagePtr[fullIdx] >= seedValueThreshold && d_visited[fullIdx] == 0)
						{
							find = true;
							d_X_new_out[curIndex1d] = tx *1.0f / width;
							d_Y_new_out[curIndex1d] = ty *1.0f / height;
							d_Z_new_out[curIndex1d] = tz *1.0f / slice;
							d_visited[fullIdx] = 1;
						}
					}

			//printf("idx: %d, tz: %d %d, ty: %d %d, tx: %d %d\n", idx, zstart, zend, ystart, yend, xstart, xend);

			/*if (find)
			{
				printf("find!, idx: %d\n", idx);
			}*/

		}
		else//����Ѿ�Ͷ���ˣ����������
		{
			return;
		}
	}


	//ͬ��
	__syncthreads();
	//����Ŀǰ PG ��֮ǰ���� PG �ڵĵ�

	for (int i = 0; i < 93; i++)
	{
		int3 offset;
		offset.x = d_offset_x[i];
		offset.y = d_offset_y[i];
		offset.z = d_offset_z[i];
		if (offset.x == 0 && offset.y == 0 && offset.z == 0) continue;

		int3 neighborIndex3d;
		neighborIndex3d.x = offset.x + curIndex3d.x;
		neighborIndex3d.y = offset.y + curIndex3d.y;
		neighborIndex3d.z = offset.z + curIndex3d.z;
		//printf("nx, ny, neiboridx: %d %d %d %d\n",nx, ny, neighborIndex2d.x, neighborIndex2d.y);
		if (neighborIndex3d.x < 0 || neighborIndex3d.x >= nx || neighborIndex3d.y < 0 || neighborIndex3d.y >= ny || neighborIndex3d.z < 0 || neighborIndex3d.z >= nz)
			continue;

		int neighborIndex1d = neighborIndex3d.x + nx * neighborIndex3d.y + nx * ny*neighborIndex3d.z;
		//printf("%d %d %d %d\n",curIndex2d.x, curIndex2d.y,  neighborIndex2d.x, neighborIndex2d.y);
		//neighborIndex ָ����neighbor����index, ��������sample��ʲô����neiborPos2d.

		//����index��Ӧ�����ʵ���꣬Ҳ����Pos
		float3 curPos3d;
		curPos3d.x = d_X_new_out[curIndex1d];
		curPos3d.y = d_Y_new_out[curIndex1d];
		curPos3d.z = d_Z_new_out[curIndex1d];
		float3 neighborPos3d;
		neighborPos3d.x = d_X_new_out[neighborIndex1d];
		neighborPos3d.y = d_Y_new_out[neighborIndex1d];
		neighborPos3d.z = d_Z_new_out[neighborIndex1d];
		//Ϊ0˵�����ھ��޵� ���ü�� ֱ��������һ��for loop����
		if (neighborPos3d.x == 0) continue;

		//����dist����r_1 r_2 �ж��Ƿ���ײ
		//dist�õ���neighbor��sample�����꣬���Ǹ��
		float dist = sqrtf(powf(curPos3d.x - neighborPos3d.x, 2) + powf(curPos3d.y - neighborPos3d.y, 2) + powf(curPos3d.z - neighborPos3d.z, 2));
		//����r_1 r_2
		//�����Ƚ�����
		//printf("%f");
		//printf("%f %f %f %d %d %f %f\n\n", r1, r2, dist, curIndex1d, neighborIndex1d, d_X_new_out[curIndex1d], d_X_new_out[neighborIndex1d]);

		//Modified by jifaley 20210618
		//impX,impY: ͨ����ǰsample������(0.12,0.65)��������importance field����ľ����±�(963*0.12,305*0.65)
		int impX = curPos3d.x * width;
		int impY = curPos3d.y * height;
		int impZ = curPos3d.z * slice;

		int curImpfullIdx = impZ * width * height + impY * width + impX;
		int curImpSmallIdx = d_compress[curImpfullIdx];
		
		int r1 = r;
		int r2 = r;


		//nimpX, nimpY: �����ھ�sample����������ھӵ�importance
		int nimpX = neighborPos3d.x * width;
		int nimpY = neighborPos3d.y * height;
		int nimpZ = neighborPos3d.z * slice;
		int neighborImpfullIdx = nimpZ * width * height + nimpY * width + nimpX;
		int neighborImpSmallIdx = d_compress[neighborImpfullIdx];


		if (dist < (r1 + r2)*1.0f / 2)
		{
			d_X_new_out[curIndex1d] = 0;
			d_Y_new_out[curIndex1d] = 0;
			d_Z_new_out[curIndex1d] = 0;
			//std::cerr << "r1: " << r1 << " r2: " << r2 << " dist: " << dist <<  ' ' << curIndex1d << ' ' << neighborIndex1d << ' ' << curPos2d.x << ' ' << curPos2d.y << ' ' << neighborPos2d.x << ' '
			//	<< neighborPos2d.y << std::endl;

			return;
		}
	}
	//printf("d_X_new_out2(%d): %f\n", idx, d_Y_new_out[curIndex1d]);

}


/*
������doPoissonSample2
���ܣ�ͨ��Poisson�����������ӵ�
���룺d_imagePtr(ԭͼ), d_imagePtr_compact(��ѹ�����ͼ)
�����seedArr(���ɸѡ������ӵ��±�)
˼·�����ɲ�������Wei et al.�� Parallel Poisson Disk Sampling ʵ�֣���δ���г���Ż�
L.-Y. Wei. Parallel poisson disk sampling. Acm Transactions On Graphics(tog), 27(3) : 1�C9, 2008.
���Ƚ������ռ仮��Ϊ��ͬ��Phase Group(PG)��ÿ��PG֮�以��Ӱ�졣
d Ϊ�����������������ռ仮�ֵĳ߶ȣ�dԽС����Խϸ
Poisson���ʣ����������֮��ľ��벻�ᳬ��ĳ����ֵ(��Ҳ�ǻ���PG��ԭ��֮һ����
*/
/*
Function��doPoissonSample2
Work��The implementaion of parallel poisson disk sampling, generate seed points for initial neuron tracing
Input��d_imagePtr(the original image), d_imagePtr_compact(the image after stream compaction)
Output��seedArr(the indices of the generated seeds in the original image)
Implementation��See L.-Y. Wei. Parallel poisson disk sampling. Acm Transactions On Graphics(tog), 27(3) : 1�C9, 2008.
This implemenation in our work are not fully optimized.
Firstly, the full space are divided in to different Phase Group(PG)s, different PGs are independent.
d: The hyper parameter, denotes the fineness of space division. A small "d" will make a fine-grained division.
Poisson property: Two random samples will have a distance larger than a fixed value.
*/

int doPoissonSample2(std::vector<int>& seedArr, dim3 center, int centerRadius, int width, int height, int slice, int newSize, uchar* d_imagePtr, uchar* d_imagePtr_compact, int* d_compress, int* d_decompress)
{
	TimerClock timer;
	timer.update();
	//����
	//ȫ����ʼ��Ϊd
	std::cout << "d = " << d << std::endl;
	float r1, r2;
	r1 = 0.1f;
	r2 = 0.1f;
	float sumValue = 0;
	float meanValue = 0;
	float value = 0;
	std::cerr << "GPU sampling malloc took me " << timer.getTimerMilliSec() << " milliseconds." << std::endl;
	timer.update();


	float radius_square = 9;
	const int radius_int = static_cast<int>(floor(sqrt(radius_square)));
	int dimension = 3;
	std::vector<offset> offset_set;
	for (int i = -radius_int; i <= radius_int; i++)
	{
		for (int j = -radius_int; j <= radius_int; j++)
		{
			for (int k = -radius_int; k <= radius_int; k++)
			{
				offset aa(k, j, i);
				if (aa.dist < radius_square)
				{
					offset_set.push_back(aa);
				}
			}
		}
	}
	//���ݾ���Բ�ĵ�Զ�����򡣺���ɸѡ��ʱ�����жϾ����Լ������neighbor�Ƿ��ͻ�����ж��ⲿ��
	//Generate the phase groups.
	std::sort(offset_set.begin(), offset_set.end(), compare_offset);
	int* offset_x = (int*)malloc(sizeof(int) * offset_set.size());
	int* offset_y = (int*)malloc(sizeof(int) * offset_set.size());
	int* offset_z = (int*)malloc(sizeof(int) * offset_set.size());

	for (int i = 0; i < offset_set.size(); i++)
	{
		offset_x[i] = offset_set[i].x;
		offset_y[i] = offset_set[i].y;
		offset_z[i] = offset_set[i].z;
	}


	//����������ÿ��level�ĸ��
	//level 0 ��ʼ��

	std::cerr << "max_level: " << max_level << std::endl;
	int num = 1;
	float r = 1;
	int nx = 1;
	int ny = 1;
	int nz = 1;
	float x0 = 1 / 2.0f;
	float y0 = 1 / 2.0f;
	float z0 = 1 / 2.0f;
	int phase0 = 1;

	//������X Y PG
	//PG == Phase Group���������򻮷ֳ�Ϊ��ͬ��Phase(����:1-9), ʹ����ͬPhase�ڵ�
	//����֮������㹻Զ���Ա㿪չ���в���Ͷ��

	/*
		�����ö�ά���������ӣ�����һ����4 * 4 = 16�����񣬻���Ϊ9��,����ÿһ���Phase���£�
		Use 2-d grid as example. Assume there is 4 * = 16 grid, we divide them into 9 groups. The phase of the groups are as follows:
		1--2--3--1
		|  |  |  |
		7--8--9--7
		|  |  |  |
		4--5--6--4
		|  |  |  |
		1--2--3--1
		�������ֳ��������񣬿��Ա�֤��������1/����2/����3....֮��ľ��붼���ڵ���3
		This division can assure that the blocks with the same phase number have a distance >= 3.
	*/

	std::vector<std::vector<float>> X;
	std::vector<std::vector<float>> Y;
	std::vector<std::vector<float>> Z;
	std::vector<std::vector<int>> PG;

	//store all the points of current level
	std::vector<float> one_X;
	std::vector<float> one_Y;
	std::vector<float> one_Z;
	std::vector<int> one_PG;
	//store each point via emplace_back func
	one_X.emplace_back(x0);
	one_Y.emplace_back(y0);
	one_Z.emplace_back(z0);
	one_PG.emplace_back(phase0);
	//store the points of current level into respective level
	X.emplace_back(one_X);
	Y.emplace_back(one_Y);
	Z.emplace_back(one_Z);
	PG.emplace_back(one_PG);


	//ÿ��level��һ��ѭ��  
	for (int level = 1; level < max_level; level++)
	{
		num = pow(8.0, level); //num��ÿ��level��������������������ά�ģ�level=1����8�����飬level=2����4*4*4=64������
		//num: the number of blocks in each level.
		r = 1 / pow(2.0, level); //r��������Ŀ��
		//r: the width of block.
		nx *= 2;
		ny *= 2;
		nz *= 2;
		x0 = 1 / 2.0f * r; //x0,y0,z0���������½��������������,ע����������Phase=1
		y0 = 1 / 2.0f * r; //x0,y0,z0 is the coordinate of the first block. Note that this block's Phase equals 1.
		z0 = 1 / 2.0f * r;
		float x, y, z;

		//clean the points remaining and store the new points
		std::vector<float>().swap(one_X);
		std::vector<float>().swap(one_Y);
		std::vector<float>().swap(one_Z);
		std::vector<int>().swap(one_PG);
		//generate & store each point via emplace_back func

		for (int k = 0; k < nz; k++)
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					x = x0 + i * r;
					y = y0 + j * r;
					z = z0 + k * r;
					one_X.emplace_back(x);
					one_Y.emplace_back(y);
					one_Z.emplace_back(z);
					phase0 = PGCompute(i + 1, j + 1, k + 1);//�±��0��ʼ ��Ҫ+1���� 
					one_PG.emplace_back(phase0);
				}
			}
		}

		//store the points of current level
		X.emplace_back(one_X);
		Y.emplace_back(one_Y);
		Z.emplace_back(one_Z);
		PG.emplace_back(one_PG);
	}

	std::cerr << "GPU sampling postprocessing 2nd stage took me " << timer.getTimerMilliSec() << " milliseconds." << std::endl;
	timer.update();

	//����Ϊֹ���������������е�Phase Group. ���濪ʼ������
	//�������������ɵ�1�׶εĳ�ʼ�㣨1���㣩�����滮��Ϊ��ͬlevel��ÿ��level����һ��level�̳�һЩ�㣬Ȼ��ͶһЩ�µ㡣

	//Till now, all of the Phase Groups are created. The sampling starts from here.
	//Sampling: First, generate the point in level 0 (only one point in the space); And the sampling process are
	//divided into multi levels. Each level will inherit some points in the last level, and generate a few new points.





	//����Xsampled �� X һģһ��
	std::vector<std::vector<float>> Xsampled(X);
	std::vector<std::vector<float>> Ysampled(Y);
	std::vector<std::vector<float>> Zsampled(Z);
	//Ŀǰ�������еĸ�㶼����ǰ�������
	//Xsampled ȫ������Ϊ0
	for (int level = 0; level < max_level; level++)
	{
		fill(Xsampled[level].begin(), Xsampled[level].end(), 0);
		fill(Ysampled[level].begin(), Ysampled[level].end(), 0);
		fill(Zsampled[level].begin(), Zsampled[level].end(), 0);
	}

	//��Xsampled�� level 0 ��Ψһһ���� ����
	Xsampled[0][0] = 1.0 / 2 + +(UniformRandom() - 0.5)*r;
	Ysampled[0][0] = 1.0 / 2 + +(UniformRandom() - 0.5)*r;
	Zsampled[0][0] = 1.0 / 2 + +(UniformRandom() - 0.5)*r;

	//��Xת��ΪX_new ÿһ����һ��X_new
	//���vector���±���PG��Ŀǰ,һ����27��Phase Group(3*3*3)
	std::vector<std::vector<float>> X_new(27);
	std::vector<std::vector<float>> Y_new(27);
	std::vector<std::vector<float>> Z_new(27);
	std::vector<int> count(27, 0);

	std::cerr << "Before random took me " << timer.getTimerMilliSec() << " milliseconds." << std::endl;
	timer.update();

	curandGenerator_t gen;//����������� ֻ����һ�ξͺ�

	uchar* d_visited;
	cudaMalloc(&d_visited, sizeof(uchar) * width * height * slice);
	cudaMemset(d_visited, 0, sizeof(uchar) * width * height * slice);


	std::cerr << "random init took me " << timer.getTimerMilliSec() << " milliseconds." << std::endl;
	timer.update();
	//cudaError_t errorCheck;
	int pg;

	float* d_X_new;
	float* d_X_new_out;
	float* d_Y_new;
	float* d_Y_new_out;
	float* d_Z_new;
	float* d_Z_new_out;
	float* d_Random;
	int* d_offset_x;
	int* d_offset_y;
	int* d_offset_z;

	//lenXYZ:��level������������
	//lenXYZ_Max:����level�����ܵ���������
	//lenXYZ:the number of blocks at the current level.
	//lenXYZ_Max:The max possible number of blocks.

	int lenXYZ_max = pow(pow(2, max_level - 1), 3) + 1;
	std::cerr << lenXYZ_max << std::endl;
	cudaMalloc((void**)&d_X_new, sizeof(float)*lenXYZ_max);
	cudaMalloc((void**)&d_Y_new, sizeof(float)*lenXYZ_max);
	cudaMalloc((void**)&d_Z_new, sizeof(float)*lenXYZ_max);
	cudaMalloc((void**)&d_X_new_out, sizeof(float)*lenXYZ_max);
	cudaMalloc((void**)&d_Y_new_out, sizeof(float)*lenXYZ_max);
	cudaMalloc((void**)&d_Z_new_out, sizeof(float)*lenXYZ_max);
	cudaMalloc((void**)&d_Random, sizeof(float)*lenXYZ_max + 2);
	cudaMalloc((void**)&d_offset_x, sizeof(int) * 93);
	cudaMalloc((void**)&d_offset_y, sizeof(int) * 93);
	cudaMalloc((void**)&d_offset_z, sizeof(int) * 93);

	std::cerr << "malloc took me " << timer.getTimerMilliSec() << " milliseconds." << std::endl;
	timer.update();

	cudaMemcpy(d_offset_x, offset_x, sizeof(int) * 93, cudaMemcpyHostToDevice);
	cudaMemcpy(d_offset_y, offset_y, sizeof(int) * 93, cudaMemcpyHostToDevice);
	cudaMemcpy(d_offset_z, offset_z, sizeof(int) * 93, cudaMemcpyHostToDevice);


	float* h_X_new = (float*)malloc(sizeof(float)*lenXYZ_max);
	float* h_Y_new = (float*)malloc(sizeof(float)*lenXYZ_max);
	float* h_Z_new = (float*)malloc(sizeof(float)*lenXYZ_max);

	std::cerr << "memcpy took me " << timer.getTimerMilliSec() << " milliseconds." << std::endl;
	timer.update();


	cudaError_t errorCheck;



	//ѭ��������Ĳ��� ����������kernel �̳нṹ��
	//The core processing (including sampling and inheriting)
	for (int level = max_level -1; level < max_level; level++) {

		//���㱾level��X_new(PG���)
		for (int n = 0; n < 27; n++)
		{
			X_new[n].clear();
			Y_new[n].clear();
			Z_new[n].clear();
		}

		//X[level].size()�ǵ�ǰlevel�ĵ�����
		for (int k = 0; k < X[level].size(); k++)
		{
			float x = X[level][k];
			float y = Y[level][k];
			float z = Z[level][k];
			//����ÿ��level�����е��Ӧ��PG
			int pg = PG[level][k];
			float R = 1.0f / pow(2.0f, level);
			X_new[pg - 1].push_back(x);
			Y_new[pg - 1].push_back(y);
			Z_new[pg - 1].push_back(z);
			//X_new_out[pg - 1].emplace_back(0);
			//Y_new_out[pg - 1].emplace_back(0);
		}


		//cout << "test" << endl;
		//�̳� ����Xsampled[level - 1] �ѵ�Ͷ�� X_new_out ��û��Ĳ���Ϊ0 kernel����0�����ж��Ƿ�Ͷ�㣩
		//��һ�������� ʹ���Ӧ�̳е������λ��
		for (int n = 0; n < 27; n++)
			count[n] = 0;

		for (int n = 0; n < Xsampled[level - 1].size(); n++)
		{
			float x = Xsampled[level - 1][n];
			float x_tmp = x;
			if (x_tmp == 0) //˵���ϸ�level�˴�û��Ͷ��
			{
				continue;
			}

			float y = Ysampled[level - 1][n];
			float y_tmp = y;
			float z = Zsampled[level - 1][n];
			float z_tmp = z;
			int nx, ny, nz;
			nx = ny = nz = (int)(pow(2.0f, level) + 0.5f);
			float blockSize = 1.0f / nx;
			//�����ϸ�level�����е��ڵ�ǰlevel��Ӧ��PG
			int ix = -1;
			int iy = -1;
			int iz = -1;
			while (x_tmp > 0)
			{
				ix++;
				x_tmp -= blockSize;
			}
			while (y_tmp > 0)
			{
				iy++;
				y_tmp -= blockSize;
			}
			while (z_tmp > 0)
			{
				iz++;
				z_tmp -= blockSize;
			}
			//index���Ǹõ��ڵ�ǰlevel���±�
			int index = ix + iy * nx + iz * nx * ny;

			pg = PG[level][index] - 1;
			Xsampled[level][index] = x;
			Ysampled[level][index] = y;
			Zsampled[level][index] = z;
			count[pg]++;
		}

		//9�� kernel parallel �����ÿ��PG��Ӧ�����в��������� ��� posXnew
		//��PG��ѭ��

		//��ǰ����device�ռ�
		int lenXYZ = 0;
		int curlenXYZ = 0;
		for (int PG = 0; PG < 27; PG++)
		{
			lenXYZ += X_new[PG].size();
		}

		//std::cerr << "Level: " << level << "lenXYZ: " << lenXYZ << std::endl;
		//����ĳ�lenXY = nx *ny Ӧ����һ���� 16 = 4* 4  = 4 + 1 +1 + ....
		//���ռ�

		//��ָ�붨λÿһ�οռ�
		float* d_cur_X_new = d_X_new;
		float* d_cur_Y_new = d_Y_new;
		float* d_cur_Z_new = d_Z_new;
		//float* d_cur_Random = d_Random;

		//curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
		//���� level PG �������������

		cudaMemcpy(d_X_new_out, &Xsampled[level][0], sizeof(float)*lenXYZ, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Y_new_out, &Ysampled[level][0], sizeof(float)*lenXYZ, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Z_new_out, &Zsampled[level][0], sizeof(float)*lenXYZ, cudaMemcpyHostToDevice);

		float * h_cur_X_new = h_X_new;
		float * h_cur_Y_new = h_Y_new;
		float * h_cur_Z_new = h_Z_new;

		int totalX = 0;
		int totalY = 0;
		int totalZ = 0;

		//������PG���������ͬһ������
		for (int PG = 0; PG < 27; PG++)
		{
			if (X_new[PG].size())
			{

				memcpy(h_cur_X_new, &X_new[PG][0], sizeof(float)*X_new[PG].size());
				memcpy(h_cur_Y_new, &Y_new[PG][0], sizeof(float)*Y_new[PG].size());
				memcpy(h_cur_Z_new, &Z_new[PG][0], sizeof(float)*Z_new[PG].size());
				h_cur_X_new += X_new[PG].size();
				h_cur_Y_new += Y_new[PG].size();
				h_cur_Z_new += Z_new[PG].size();
				totalX += X_new[PG].size();
				totalY += Y_new[PG].size();
				totalZ += Z_new[PG].size();
			}
		}

		cudaMemcpy(d_X_new, h_X_new, sizeof(float) * totalX, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Y_new, h_Y_new, sizeof(float) * totalY, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Z_new, h_Z_new, sizeof(float) * totalZ, cudaMemcpyHostToDevice);

		thrust::counting_iterator<unsigned int> index_sequence_begin(10007);
		int seedValueThreshold = 15;

		for (int K = 0; K < 1; K++)
		{

			thrust::transform(thrust::device, index_sequence_begin, index_sequence_begin + (lenXYZ + 2), d_Random, psrngen(0.0f, 1.0f));
			//curandGenerateUniform(gen, d_Random, lenXYZ + 2);
			d_cur_Y_new = d_Y_new;
			d_cur_X_new = d_X_new;
			d_cur_Z_new = d_Z_new;

			for (int PG = 0; PG < 27; PG++)
			{
				if (X_new[PG].size()) {

					dim3 grid((X_new[PG].size() + 1024 - 1) / 1024, 1, 1);
					dim3 block(1024, 1, 1);

					//std::cerr << "Level: " << level << " PG: " << PG << " grid: " << grid.x << " X_New[PG].size " << X_new[PG].size() << std::endl;

					samplingKernel2 << <grid, block >> > (d_cur_X_new, d_cur_Y_new, d_cur_Z_new,
						d_X_new_out, d_Y_new_out, d_Z_new_out,
						d_Random,
						X_new[PG].size(), 1.0 / pow(2, level),
						PG, level,
						d_offset_x, d_offset_y, d_offset_z,
					  d_imagePtr, d_visited, d_compress, width, height, slice, seedValueThreshold);
					//__syncthreads();
					//ÿ��PG�ĵ���ú󣬴���host
					errorCheck = cudaGetLastError();
					if (errorCheck != cudaSuccess) {
						std::cerr << "Error During Sample" << cudaGetErrorString(errorCheck) << std::endl;
						system("pause");
						return -1;
					}
				}
				//ָ���ƶ����¶��ڴ��׵�ַ
				d_cur_X_new += X_new[PG].size();
				d_cur_Y_new += Y_new[PG].size();
				d_cur_Z_new += Z_new[PG].size();
			}
		}

		cudaMemcpy(&Xsampled[level][0], d_X_new_out, sizeof(float)*lenXYZ, cudaMemcpyDeviceToHost);
		cudaMemcpy(&Ysampled[level][0], d_Y_new_out, sizeof(float)*lenXYZ, cudaMemcpyDeviceToHost);
		cudaMemcpy(&Zsampled[level][0], d_Z_new_out, sizeof(float)*lenXYZ, cudaMemcpyDeviceToHost);

		//free�������ڴ�
		std::cerr << "Level " << level << " GPU sampling took me " << timer.getTimerMilliSec() << " milliseconds." << std::endl;
		timer.update();


	}


	std::cerr << "GPU sampling took me " << timer.getTimerMilliSec() << " milliseconds." << std::endl;
	timer.update();

	//�������ɸѡ
	//Filter the samples

	int lastlevel = Xsampled.size() - 1;
	int sampleNum = Xsampled[lastlevel].size();


	filterPoissonSample_gpu(seedArr, center, centerRadius, sampleNum,
		width, height, slice, d_imagePtr, d_X_new_out, d_Y_new_out, d_Z_new_out);

	std::cerr << "GPU filtering took me " << timer.getTimerMilliSec() << " milliseconds." << std::endl;
	timer.update();


	cudaFree(d_X_new);
	cudaFree(d_Y_new);
	cudaFree(d_Z_new);
	cudaFree(d_X_new_out);
	cudaFree(d_Y_new_out);
	cudaFree(d_Z_new_out);
	cudaFree(d_Random);
	cudaFree(d_offset_x);
	cudaFree(d_offset_y);
	cudaFree(d_offset_z);
	cudaFree(d_visited);


	free(offset_x);
	free(offset_y);
	free(offset_z);
	free(h_X_new);
	free(h_Y_new);
	free(h_Z_new);

	std::cerr << "Free took me " << timer.getTimerMilliSec() << " milliseconds." << std::endl;
	timer.update();


	return 0;
}
