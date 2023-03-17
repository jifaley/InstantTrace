#include "mst.h"
#include "fastmarching.h"
#include "pruning.h"
using namespace std;


int minKey(float key[], uchar mstSet[], int v)
{
	// Initialize min value 
	int min = INT_MAX, min_index;
	min_index = -1;

	for (int i = 0; i < v; i++)
		if (mstSet[i] == FAR && key[i] < min)
			min = key[i], min_index = i;

	return min_index;
}


void primPruneNode(uchar* d_imagePtr_compact, std::vector<int> & seedArr, short int* d_seedNumberPtr, int* d_compress, int* d_decompress, uchar* d_radiusMat_compact, uchar* d_activeMat_compact, int* d_parentPtr_compact, int width, int height, int slice, int newSize)
{




}




// Function to construct and print MST for a graph represented using adjacency 
// matrix representation 
void primMST(uchar* d_imagePtr_compact, std::vector<int> & seedArr, short int* d_seedNumberPtr, int* d_compress, int* d_decompress, uchar* d_radiusMat_compact, uchar* d_activeMat_compact, int* d_parentPtr_compact, int width , int height, int slice, int newSize)
{
	//int parent[v]; // Array to store constructed MST 
	//int key[v]; // Key values used to pick minimum weight edge in cut 
	//bool mstSet[v]; // To represent set of vertices not yet included in MST 
	int v = newSize;
	int n = width * height * slice;
	int seedCount = seedArr.size();

	uchar* h_imagePtr_comapct = (uchar*)malloc(sizeof(uchar) * v);

	int * h_compress = (int*)malloc(sizeof(int) * n);
	int * h_decompress = (int*)malloc(sizeof(int) * v);
	uchar * h_radiusMat_compact = (uchar*)malloc(sizeof(uchar) * v);
	short int * h_seedNumberPtr = (short int*)malloc(sizeof(short int) * v);

	std::cerr << "primMST: malloc finished" << std::endl;

	cudaMemcpy(h_compress, d_compress, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_decompress, d_decompress, sizeof(int) * v, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_radiusMat_compact, d_radiusMat_compact, sizeof(uchar) * v, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_imagePtr_comapct, d_imagePtr_compact, sizeof(uchar) * v, cudaMemcpyDeviceToHost);
	
	std::cerr << "primMST: memcpy finished" << std::endl;


	int* parent = (int*)malloc(sizeof(int) * v * 2); //注意：两个parent
	float * key = (float*)malloc(sizeof(float) * v);
	uchar * mstSet = (uchar*)malloc(sizeof(bool) * v);

	// Initialize all keys as INFINITE 
	for (int i = 0; i < v; i++)
		key[i] = INT_MAX, mstSet[i] = FAR, h_seedNumberPtr[i] = 0;
	for (int i = 0; i < v * 2; i++)
		parent[i] = -1;

	int centerFullIdx, centerSmallIdx;

	for (int i = 0; i < seedCount; i++)
	{
		centerFullIdx = seedArr[i];
		centerSmallIdx = h_compress[centerFullIdx];
		std::cerr << "CenterSmallIDx: " << centerSmallIdx << std::endl;
		// Always include first 1st vertex in MST. 
		key[centerSmallIdx] = 0;     // Make key 0 so that this vertex is picked as first vertex 
		parent[centerSmallIdx] = centerSmallIdx; // First node is always root of MST 
		//Hint: The seed number are started from 1, not 0. The 0th seed is a dummy seed.
		h_seedNumberPtr[centerSmallIdx] = i + 1;
	}


	int intersect = 0;

	// The MST will have v vertices 
	for (int count = 0; count < v - 1; count++)
	{
		// Pick thd minimum key vertex from the set of vertices 
		// not yet included in MST 
		if (count % 100 == 0)
			std::cerr << count << std::endl;

		int smallIdx = minKey(key, mstSet, v);
		
		if (smallIdx == -1) break;

		// Add the picked vertex to the MST Set 
		mstSet[smallIdx] = ALIVE;

		int fullIdx = h_decompress[smallIdx];

		int z = fullIdx / (width * height);
		int y = fullIdx % (width * height) / width;
		int x = fullIdx % width;

		for (int k = max(z-1, 0); k <= min(z+1, slice-1); k++)
			for (int i = max(y-1, 0); i <= min(y+1, height -1); i++)
				for (int j = max(x - 1, 0); j <= min(x + 1, width - 1); j++)
				{
					if (k == z && i == y && j == x) continue;
					if ((z - k) * (z - k) + (y - i) * (y - i) + (x - j) * (x - j) != 1) continue;
					int neighborfullIdx = k * width * height + i * width + j;
					int neighborSmallIdx = h_compress[neighborfullIdx];

					//if (neighborSmallIdx > 0 && mstSet[neighborSmallIdx] == FAR
					//	&& 1.0 / h_radiusMat_compact[neighborSmallIdx] < key[neighborSmallIdx])
					//	parent[neighborSmallIdx] = smallIdx, key[neighborSmallIdx] = 1.0 / h_radiusMat_compact[neighborSmallIdx], h_seedNumberPtr[neighborSmallIdx] = h_seedNumberPtr[smallIdx];
					//else
					//	if (neighborSmallIdx > 0 && mstSet[neighborSmallIdx] == ALIVE
					//		&& 1.0 / h_radiusMat_compact[neighborSmallIdx] < key[neighborSmallIdx] && parent[neighborSmallIdx] != smallIdx)
					//		intersect += 1, parent[neighborSmallIdx + v] = parent[neighborSmallIdx], parent[neighborSmallIdx] = smallIdx, key[neighborSmallIdx] = 1.0 / h_radiusMat_compact[neighborSmallIdx], h_seedNumberPtr[neighborSmallIdx] = h_seedNumberPtr[smallIdx];
				
					if (neighborSmallIdx >= 0)
					{
						uchar neighborValue = h_imagePtr_comapct[neighborSmallIdx];
						if (neighborValue == 0) continue; //这个==0是gwdt引起的

						int curSeed = h_seedNumberPtr[smallIdx];
						int neighborSeed = h_seedNumberPtr[neighborSmallIdx];

						float deltaDist;

						if (h_radiusMat_compact[neighborSmallIdx] > 1e-5)
							deltaDist = 1.0 / h_radiusMat_compact[neighborSmallIdx];
						else
							deltaDist = 1024;
						

						//float EuclidDist = sqrtf((z - k) * (z - k) + (y - i) * (y - i) + (x - j) * (x - j));
						//uchar curValue = h_imagePtr_comapct[smallIdx];
						//uchar neighborValue = h_imagePtr_comapct[neighborSmallIdx];
						//deltaDist = gwdtFunc(EuclidDist, curValue, neighborValue);

						//如果找到了更近的路径
						if (deltaDist  < key[neighborSmallIdx])
						{
							key[neighborSmallIdx] = deltaDist;

							if (curSeed != neighborSeed)
							{
								//如果来自不同seed，记录第二parent
								parent[neighborSmallIdx + v] = parent[neighborSmallIdx];
								h_seedNumberPtr[neighborSmallIdx] = curSeed;
								if (neighborSeed != 0) intersect++;
							}
							//mstSet[neighborSmallIdx] = FAR;
							parent[neighborSmallIdx] = smallIdx;
						}	
					}
				}


		/*

		// Update key value and parent index of the adjacent vertices of 
		// the picked vertex. Consider only those vertices which are not yet 
		// included in MST 
		for (int i = 0; i < v; i++)

			// graph[u][v] is non zero only for adjacent vertices of m 
			// mstSet[v] is false for vertices not yet included in MST 
			// Update the key only if graph[u][v] is smaller than key[v] 
			if (graph[u][i] && mstSet[i] == false && graph[u][i] < key[i])
				parent[i] = u, key[i] = graph[u][i];
		*/
	}

	int nozero_count = 0;
	for (int i = 0; i < v; i++)
	{
		if (parent[i] != -1)
			nozero_count++;
	}

	std::cerr << "nozero count: " << nozero_count << std::endl;
	std::cerr << "intersect count:" << intersect << std::endl;


	std::cerr << "primMST: calc finished" << std::endl;
	cudaMemcpy(d_parentPtr_compact, parent, sizeof(int) * v * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_seedNumberPtr, h_seedNumberPtr, sizeof(short int) * v, cudaMemcpyHostToDevice);
	cudaMemcpy(d_activeMat_compact, mstSet, sizeof(uchar) * v, cudaMemcpyHostToDevice);

	free(parent);
	free(key);
	free(mstSet);

	free(h_compress);
	free(h_decompress);
	free(h_radiusMat_compact);
	free(h_seedNumberPtr);
	free(h_imagePtr_comapct);

	std::cerr << "primMST: free finished" << std::endl;
	// print the constructed MST 
}


void mst(uchar* h_imagePtr_compact, std::vector<int>& seedArr, int* h_parentPtr_compact, short int* h_seedNumberPtr, uchar* h_activeMat_compact, int* h_childNumMat, int width, int height, int slice, int newSize)
{
	int seedNum = seedArr.size();
	int currentSeed = 0;
	int seedCount = 0;
	while (seedCount < seedNum)
	{
		currentSeed = seedArr[seedCount]; //??small or full??
		seedCount = seedCount + 1;



	}
}