#include "loadTiff.h"


void loadTiff(const char* file, std::vector<cv::Mat> &buffer, int *size)
{
	TIFF *tif = TIFFOpen(file, "r");      //使用TIFFOpen函数以只读形式打开图像。
	if (tif == nullptr)
	{
		std::cerr << "读入图像路径错误,请重新确认";
		return;
	}

	int width, height;

	//-------------获取单帧图像的长高
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
	//TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &channel);     //------获取通道数

	//------------ 获取图片帧数
	int nTotalFrame = TIFFNumberOfDirectories(tif);
	std::cerr << "width: " << width   << std::endl;
	std::cerr << "Slice: " << nTotalFrame << std::endl;
	std::cerr << "height: " << height << std::endl;


	//---------------获取每一帧的像素点数目
	int stripSize = TIFFStripSize(tif);

	//---------------------申请单帧图像所需的内存空间；
	uint32* count = new uint32[height*width];
	//uint32* count = new uint32[stripSize]; 


	for (int s = 0; s < nTotalFrame; s++)
	{

		//---------------------建立单张画布；
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

void saveTiff(const char *path, uchar *buffer, int *size)
{
	int width = size[0];
	int height = size[1];
	int slice = size[2];

	TIFF* out = TIFFOpen(path, "w");
	if (out)
	{
		int N_size = 0;
		size_t nCur = 0;
		//UChar den = (sizeof(T) == 1) ? 1 : 4;
		do {
			TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
			TIFFSetField(out, TIFFTAG_PAGENUMBER, slice);
			TIFFSetField(out, TIFFTAG_IMAGEWIDTH, (uint32)width);
			TIFFSetField(out, TIFFTAG_IMAGELENGTH, (uint32)height);
			//TIFFSetField(out, TIFFTAG_RESOLUTIONUNIT, 2);
			/*TIFFSetField(out, TIFFTAG_YRESOLUTION, 196.0f);
			TIFFSetField(out, TIFFTAG_XRESOLUTION, 204.0f);*/
			TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
			// 
			TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);    //根据图像位深填不同的值
			TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
			TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
			TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
			TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
			TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, height);


			for (int m = 0; m < height; m++)
			{
				TIFFWriteScanline(out, &buffer[N_size + width * m], m, 0);
			}
			//TIFFWriteEncodedStrip(out, 0, &buffer[N_size], width * height);      //另一种写入方法

			++nCur;
			N_size = N_size + width * height;
		} while (TIFFWriteDirectory(out) && nCur < slice);
		TIFFClose(out);

		std::cerr << "save over" << std::endl;
	}
}


//读取元素
/*

uint32* rowPoint2Src = count + (height - 1)*width;     //构建一个指向最后一行的第一个元素。注意这里，是从最后一行开始读入的

for (int i = 0; i <height; i++)
{
	uint32* colPoint2Src = rowPoint2Src;

	for (int j = 0; j <width; j++)
	{

	MatImage.at<uchar>(i, j) = (uchar)TIFFGetG(*colPoint2Src);      //按照像素点读取
	colPoint2Src++;

	}
	rowPoint2Src -= width;
}


*/


//main函数举例
/*
void main()

{
	cv::Mat loadImage;
	std::vector<cv::Mat> imageSeries;
	std::string filename = "neuron01.tif";
	std::vector<int>size(3);
	loadTiff(filename.c_str(), imageSeries,&size[0]);
	std::cout << size[0] << ' '<< size[1]  << ' ' << size[2] << std::endl;
	//963*305*140
	cv::Mat OutImage = imageSeries[139];
	imshow("test", OutImage);
	std::cout << (float)imageSeries[60].at<uchar>(304, 962); 
	cv::imwrite("testTiff.jpg", imageSeries[60]);
	cv::waitKey(0);
	OutImage.release();
}
*/