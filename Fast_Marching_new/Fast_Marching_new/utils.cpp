#include "utils.h"
#include <io.h>



uchar* loadImage(const std::string inputName, int* imageShape)
{
	std::vector<cv::Mat> imageSeries;
	loadTiff(inputName.c_str(), imageSeries, imageShape);
	int width = imageShape[0]; //963
	int height = imageShape[1]; //305
	int slice = imageShape[2]; //140
	std::cerr << width << ' ' << height << ' ' << slice << std::endl;
	uchar* imagePtr = (uchar*)malloc(sizeof(uchar) * width * height * slice);
	uchar* currentPos = imagePtr;
	for (int i = 0; i < slice; i++)
	{
		memcpy(currentPos, imageSeries[i].ptr(), sizeof(uchar) * width * height);
		currentPos += (width * height);
	}
	assert(imagePtr != NULL);
	return imagePtr;
}


void getFiles(std::string path, std::vector<std::string>& files, std::vector<std::string>& names)
{
	//�ļ������win10��long long��win7��long�Ϳ�����
	long long hFile = 0;
	//�ļ���Ϣ 
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//�����Ŀ¼,����֮ //�������,�����б� 
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					getFiles(p.assign(path).append("\\").append(fileinfo.name),files, names);
				}
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name)); 
					names.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}