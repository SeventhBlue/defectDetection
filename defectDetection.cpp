#include "defectDetection.h"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

std::vector<DefectDetectSt> biob(cv::Mat& src, int minArea, int maxArea) {
	cv::Mat gray;
	if (src.channels() == 3) {
		cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	}
	else {
		gray = src;
	}
	//cv::imshow("gray", gray);

	gray = 255 - gray;
	cv::Mat thres;
	cv::threshold(gray, thres, 230, 255, cv::THRESH_BINARY);
	//cv::imshow("thres", thres);
	thres = 255 - thres;
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(thres, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	cv::Rect boundRect;
	for (int i = 0; i < contours.size(); ++i) {
		boundRect = cv::boundingRect((cv::Mat)contours[i]);
		if (((boundRect.width * boundRect.height) > 600) && (abs(boundRect.width - boundRect.height) < 5)) {
			int x = boundRect.x + boundRect.width / 2;
			int y = boundRect.y + boundRect.height / 2;
			int rad = boundRect.height / 2;
			cv::circle(thres, cv::Point(x, y), rad, cv::Scalar(255, 255, 255), 2);
		}
	}
	//cv::imshow("thres", thres);

	thres = 255 - thres;
	std::vector<DefectDetectSt> ret;
	cv::Mat connect, stats, centroids;
	int areaNum = cv::connectedComponentsWithStats(thres, connect, stats, centroids, 8, CV_16U);
	for (int i = 1; i < areaNum; ++i) {
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		if (area >= minArea && area < maxArea) {
			DefectDetectSt dds;
			dds.centerX = centroids.at<double>(i, 0);
			dds.centerY = centroids.at<double>(i, 1);
			
			dds.rect.x = stats.at<int>(i, cv::CC_STAT_LEFT);
			dds.rect.y = stats.at<int>(i, cv::CC_STAT_TOP);
			dds.rect.width = stats.at<int>(i, cv::CC_STAT_WIDTH);
			dds.rect.height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
			dds.area = area;
			ret.push_back(dds);

			cv::circle(src, cv::Point(dds.centerX, dds.centerY), 2, cv::Scalar(0, 255, 0), 2, 8, 0);
			cv::rectangle(src, dds.rect, cv::Scalar(0, 0, 255), 1, 8, 0);
		}
	}
}