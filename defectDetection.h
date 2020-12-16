#pragma once
#include <iostream>
#include "opencv2/imgproc.hpp"

struct DefectDetectSt {
	cv::Rect rect;
	int area;
	int centerX;
	int centerY;
};

std::vector<DefectDetectSt> biob(cv::Mat& src, int minArea, int maxArea);