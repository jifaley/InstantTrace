#include "mergeSegments.h"
#include "TimerClock.hpp"



int getfather(std::vector<int>& fa, int x)
{
	//std::cerr << x << std::endl;
	if (fa[x] == x) return x;
	return fa[x] = getfather(fa, fa[x]);
}

void merge(std::vector<int>& fa, std::vector<int>& seedRadius, int x, int y)
{
	int fa_x = getfather(fa, x);
	int fa_y = getfather(fa, y);
	//�뾶���Ϊ���ף���ͬ�����
	//���С��Ϊ����

	if (seedRadius[fa_x] > seedRadius[fa_y])
	{
		fa[fa_y] = fa_x;
	}
	else if (seedRadius[fa_x] < seedRadius[fa_y])
	{
		fa[fa_x] = fa_y;
	}
	else if (seedRadius[fa_x] == seedRadius[fa_y])
	{
		if (fa_x < fa_y)
			fa[fa_y] = fa[x];
		else
			fa[fa_x] = fa[y];
	}
}

inline float gIFunc(float value, float maxvalue = 255, float lambda = 10)
{
	return exp(lambda * (1 - value * 1.0 / maxvalue) * (1 - value * 1.0 / maxvalue));
	//gI(x) = exp(\lambda_i * (1 - I(x)/Imax) ** 2)
}

inline float gwdtFunc(float dist, float value1, float value2)
{
	return dist * (gIFunc(value1) + gIFunc(value2)) / 2;
	//e(x,y) = |x-y| * (gI(x) + gI(y)) / 2
}


void mergeSegments(uchar* imagePtr, std::vector<int>& seedArr, int width, int height, int slice, uchar* statusMat, int* parentMat, short int* parentRootMat, uchar* childNumMat, uchar* radiusMat)
{
	//Adding Pruning Merge 20211030
	TimerClock timer;
	timer.update();
	std::vector<int> intersectArr;
	std::vector<int> seedRadiusArr(seedArr.size() + 1);

	for (int it = 1; it < seedArr.size() + 1; it++)
	{
		int curIndex = seedArr[it - 1];
		seedRadiusArr[it] = radiusMat[curIndex];
	}

	int ra, rb;
	int countIfExistBothRoot = 0;
	//�ж�һ��ÿ�����㸽���Ƿ���������������
	//Ҳ���ڸ��µ�ʱ���Ҫͬʱ���µڶ���root(����parent�ĵڶ�Root����?)

	for (int it = 0; it < width * height * slice; it++)
	{
		ra = parentRootMat[it];
		rb = parentRootMat[it + width * height * slice];

		if (ra != 0 && rb != 0 && ra != rb)
		{
			int parentIdx2 = parentMat[it + width * height * slice];
			if (parentIdx2 != -1 && parentIdx2 != it && parentRootMat[parentIdx2] == rb)
			{
					intersectArr.push_back(it);	
			}
		}
	}

	std::cerr << "InterSect Size: " << intersectArr.size() << std::endl;
	std::cerr << "InterSect Finding cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();


	int countValidInterSect = 0;
	std::vector<int> disjointSet(seedArr.size() + 1, 0);
	for (int it = 0; it < disjointSet.size(); it++)
		disjointSet[it] = it;

	std::vector<bool> ifInterSectValid(intersectArr.size(), false);


	for (int it = 0; it < intersectArr.size(); it++)
	{
		int curIdx = intersectArr[it];
		int parentIdx = parentMat[curIdx];
		int parent2Idx = parentMat[curIdx + width * height * slice];

		int curRoot = parentRootMat[curIdx];
		int parentRoot = parentRootMat[parentIdx];
		int parent2Root = parentRootMat[parent2Idx];

		bool checkIfValidInterSect = true;

		int curIdxTemp = curIdx;
		while (parentMat[curIdxTemp] != curIdxTemp)
		{
			int curRootTemp = parentRootMat[curIdxTemp];
			if (curRootTemp != curRoot)
			{
				checkIfValidInterSect = false;
				break;
			}
			curIdxTemp = parentMat[curIdxTemp];
		}

		curIdxTemp = parent2Idx;
		while (parentMat[curIdxTemp] != curIdxTemp)
		{
			int curRootTemp = parentRootMat[curIdxTemp];
			if (curRootTemp != parent2Root)
			{
				checkIfValidInterSect = false;
				break;
			}
			curIdxTemp = parentMat[curIdxTemp];
		}
		if (checkIfValidInterSect)
		{
			ifInterSectValid[it] = true;
			countValidInterSect++;
		}
		else
		{
			ifInterSectValid[it] = false;
			continue;
		}
	}
	std::cerr << "Valid InterSect: " << countValidInterSect << std::endl;
	std::cerr << "InterSect Chekcing cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();



	for (int it = 0; it < intersectArr.size(); it++)
	{
		if (ifInterSectValid[it] == false)
			continue;
		int curIdx = intersectArr[it];
		int parentIdx = parentMat[curIdx];
		int parent2Idx = parentMat[curIdx + width * height * slice];

		int curRoot = parentRootMat[curIdx];
		int parentRoot = parentRootMat[parentIdx];
		int parent2Root = parentRootMat[parent2Idx];

		//�����ڱ�֤valid��
		int father1 = getfather(disjointSet, curRoot);
		int father2 = getfather(disjointSet, parent2Root);
		int prevIdxTemp, nextIdxTemp, curIdxTemp;
		//father��һ����Ҫ�ϲ�
		if (father1 != father2)
		{
			//std::cerr << "Merge:" << father1 << ' ' << father2 << std::endl;
			//С�ĵ���,��parent2��ʼ,�𽥷�����,parnet2��parent��Ϊcur
			if (father1 < father2)
			{
				curIdxTemp = parent2Idx;
				prevIdxTemp = curIdx;
				while (parentMat[curIdxTemp] != curIdxTemp) //|| parentRootMat[curIdxTemp] != father2)
				{

					nextIdxTemp = parentMat[curIdxTemp];

					//1.�޸�
					parentMat[curIdxTemp] = prevIdxTemp;
					//parentMat[curIdxTemp + width * height * slice] = -1;
					//2.ǰ��
					prevIdxTemp = curIdxTemp;
					curIdxTemp = nextIdxTemp;
				}
				//ȷ���ߵ�root��root�����parent�յ����޸�
				if (parentMat[curIdxTemp] == curIdxTemp && curIdxTemp != prevIdxTemp)
				{
					parentMat[curIdxTemp] = prevIdxTemp;
				}
			}
			//С�ĵ���,��cur��ʼ,�𽥷�����,cur��parent��Ϊparent2
			else if (father1 > father2)
			{
				curIdxTemp = curIdx;
				prevIdxTemp = parent2Idx;
				while (parentMat[curIdxTemp] != curIdxTemp) //|| parentRootMat[curIdxTemp] != father2)
				{
					nextIdxTemp = parentMat[curIdxTemp];

					//1.�޸�
					parentMat[curIdxTemp] = prevIdxTemp;
					//parentMat[curIdxTemp + width * height * slice] = -1;
					//2.ǰ��
					prevIdxTemp = curIdxTemp;
					curIdxTemp = nextIdxTemp;
				}
				//ȷ���ߵ�root��root�����parent�յ����޸�
				if (parentMat[curIdxTemp] == curIdxTemp && curIdxTemp != prevIdxTemp)
				{
					parentMat[curIdxTemp] = prevIdxTemp;
				}
			}

			merge(disjointSet, seedRadiusArr, father1, father2);
		}
	}
	//Ends

	std::cerr << "Merge Reverse Complete" << std::endl;
	std::cerr << "Merging cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();

	////�鿴�޸����parent����û�зɱ�
	//for (int i = 0; i < width * height * slice; i++)
	//{
	//	int parent = parentMat[i];
	//	if (parent == -1) continue;
	//	int3 curPos;
	//	curPos.z = i / (width * height);
	//	curPos.y = i % (width * height) / width;
	//	curPos.x = i % width;

	//	int3 parentPos;
	//	parentPos.z = parent / (width * height);
	//	parentPos.y = parent % (width * height) / width;
	//	parentPos.x = parent % width;

	//	float EuclidDist = sqrt((curPos.x - parentPos.x) * (curPos.x - parentPos.x)
	//		+ (curPos.y - parentPos.y) * (curPos.y - parentPos.y)
	//		+ (curPos.z - parentPos.z) * (curPos.z - parentPos.z));
	//	if (EuclidDist > 1.8)
	//	{
	//		std::cerr << "current: " << curPos.x << ' ' << curPos.y << ' ' << curPos.z << "     "
	//			<< "parent: " << parentPos.x << ' ' << parentPos.y << ' ' << parentPos.z << std::endl;
	//	}
	//}
	////�����޸���parent�������޸�dist
	//std::cerr << "Merge Checking cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	//timer.update();



	//������ͳ��childNum
	memset(childNumMat, 0, sizeof(uchar) * width * height * slice);
	for (int i = 0; i < width * height * slice; i++)
	{
		int parent = parentMat[i];
		if (parent == -1) continue;
		if (parent != i)
			childNumMat[parent]++;
	}

	std::cerr << "Renew ChildNum cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();
}