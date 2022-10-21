#include "loadTiff.h"


//Loading *.tif file. User can change this to faster image loading techniques.
void loadTiff(const char* file, std::vector<cv::Mat> &buffer, int *size)
{
	TIFF *tif = TIFFOpen(file, "r");
	if (tif == nullptr)
	{
		std::cerr << "读入图像路径错误,请重新确认";
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

		TIFFSetDirectory(tif, s);       //-------选中第s帧

		TIFFReadRGBAImage(tif, width, height, count, 0);       //将第s帧的内容传递到count中；默认参数选择是0，所以是逆序读入的。可以使用ORIENTATION_TOPLEFT作为参数，从左上角开始
		uint32* rowPoint2Src = count + (height - 1)*width;     //构建一个指向最后一行的第一个元素。注意这里，是从最后一行开始读入的

		for (int i = 0; i < height; i++)
		{
			uint32* colPoint2Src = rowPoint2Src;

			for (int j = 0; j < width; j++)
			{

				MatImage.at<uchar>(i, j) = (uchar)TIFFGetG(*colPoint2Src);      //按照像素点读取
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