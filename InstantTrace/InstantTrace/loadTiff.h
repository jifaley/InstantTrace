#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <tiffio.h>
#include <iostream>

void loadTiff(const char* file, std::vector<cv::Mat> &buffer, int *size);
