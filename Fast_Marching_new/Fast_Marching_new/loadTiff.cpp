#include "loadTiff.h"


void loadTiff(const char* file, std::vector<cv::Mat> &buffer, int *size)
{
	TIFF *tif = TIFFOpen(file, "r");      //ʹ��TIFFOpen������ֻ����ʽ��ͼ��
	if (tif == nullptr)
	{
		std::cerr << "����ͼ��·������,������ȷ��";
		return;
	}

	int width, height;

	//-------------��ȡ��֡ͼ��ĳ���
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
	//TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &channel);     //------��ȡͨ����

	//------------ ��ȡͼƬ֡��
	int nTotalFrame = TIFFNumberOfDirectories(tif);
	std::cerr << "width: " << width   << std::endl;
	std::cerr << "Slice: " << nTotalFrame << std::endl;
	std::cerr << "height: " << height << std::endl;


	//---------------��ȡÿһ֡�����ص���Ŀ
	int stripSize = TIFFStripSize(tif);

	//---------------------���뵥֡ͼ��������ڴ�ռ䣻
	uint32* count = new uint32[height*width];
	//uint32* count = new uint32[stripSize]; 


	for (int s = 0; s < nTotalFrame; s++)
	{

		//---------------------�������Ż�����
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
			TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);    //����ͼ��λ���ͬ��ֵ
			TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
			TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
			TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
			TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
			TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, height);


			for (int m = 0; m < height; m++)
			{
				TIFFWriteScanline(out, &buffer[N_size + width * m], m, 0);
			}
			//TIFFWriteEncodedStrip(out, 0, &buffer[N_size], width * height);      //��һ��д�뷽��

			++nCur;
			N_size = N_size + width * height;
		} while (TIFFWriteDirectory(out) && nCur < slice);
		TIFFClose(out);

		std::cerr << "save over" << std::endl;
	}
}


//��ȡԪ��
/*

uint32* rowPoint2Src = count + (height - 1)*width;     //����һ��ָ�����һ�еĵ�һ��Ԫ�ء�ע������Ǵ����һ�п�ʼ�����

for (int i = 0; i <height; i++)
{
	uint32* colPoint2Src = rowPoint2Src;

	for (int j = 0; j <width; j++)
	{

	MatImage.at<uchar>(i, j) = (uchar)TIFFGetG(*colPoint2Src);      //�������ص��ȡ
	colPoint2Src++;

	}
	rowPoint2Src -= width;
}


*/


//main��������
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