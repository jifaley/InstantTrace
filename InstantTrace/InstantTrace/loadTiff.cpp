#include "loadTiff.h"


//Loading *.tif file. User can change this to faster image loading techniques.
void loadTiff(const char* file, std::vector<cv::Mat> &buffer, int *size)
{
	TIFF *tif = TIFFOpen(file, "r");
	if (tif == nullptr)
	{
		std::cerr << "����ͼ��·������,������ȷ��";
		return;
	}

	int width, height;
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

	int nTotalFrame = TIFFNumberOfDirectories(tif);
	std::cerr << "width: " << width   << std::endl;
	std::cerr << "Slice: " << nTotalFrame << std::endl;
	std::cerr << "height: " << height << std::endl;

	uint32* count = new uint32[height*width];
	for (int s = 0; s < nTotalFrame; s++)
	{
		cv::Mat MatImage(height, width, CV_8UC1, cv::Scalar::all(0));

		TIFFSetDirectory(tif, s);       //-------ѡ�е�s֡

		TIFFReadRGBAImage(tif, width, height, count, 0);       //����s֡�����ݴ��ݵ�count�У�Ĭ�ϲ���ѡ����0���������������ġ�����ʹ��ORIENTATION_TOPLEFT��Ϊ�����������Ͻǿ�ʼ
		uint32* rowPoint2Src = count + (height - 1)*width;     //����һ��ָ�����һ�еĵ�һ��Ԫ�ء�ע������Ǵ����һ�п�ʼ�����

		for (int i = 0; i < height; i++)
		{
			uint32* colPoint2Src = rowPoint2Src;

			for (int j = 0; j < width; j++)
			{

				MatImage.at<uchar>(i, j) = (uchar)TIFFGetG(*colPoint2Src);      //�������ص��ȡ
				colPoint2Src++;

			}
			rowPoint2Src -= width;
		}

		buffer.push_back(MatImage);
		MatImage.release();
	}

	TIFFClose(tif);
	delete[] count;

	size[0] = width;
	size[1] = height;
	size[2] = nTotalFrame;
}