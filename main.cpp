#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


#include "crackDetection.h"
#include "crackInfo.h"
#include "defectDetection.h"

void defectDetection() {
	cv::VideoCapture cap;
	cap.open("123.avi");
	cv::Mat frame;
	while (cv::waitKey(1) < 0) {
		cap >> frame;
		if (frame.empty()) {
			cv::waitKey();
			break;
		}

		double start_time = (double)cv::getTickCount();

		biob(frame, 20, 2000);

		double end_time = (double)cv::getTickCount();
		double fps = cv::getTickFrequency() / (end_time - start_time);
		double spend_time = (end_time - start_time) / cv::getTickFrequency();
		std::string FPS = "FPS:" + cv::format("%.2f", fps) + "  spend time:" + cv::format("%.2f", spend_time * 1000) + "ms";
		putText(frame, FPS, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
		imshow("ret", frame);
	}

	cv::destroyAllWindows();

}
void crackDetection() {
	cv::Mat srcImg = cv::imread("12.png");

	cv::Mat destImg;
	CrackDetection::detect(srcImg, destImg);

	cv::imshow("srcImg", srcImg);
	cv::imshow("result", destImg);
	cv::waitKey(0);
	//cv::imwrite("result.png", destImg);
}

int main() {

	crackDetection();

	defectDetection();
	
	return 0;
}