#pragma once

#include <deque>
#include <stack>
#include <fstream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "crackInfo.h"

class CrackDetection
{
public:
	CrackDetection();
	~CrackDetection();
	/* 增加对比度 */
	static void addContrast(cv::Mat &srcImg);
	/* 检测连通域，并删除不符合条件的连通域 */
	static void findConnectedDomain(cv::Mat &srcImg, std::vector<std::vector<cv::Point>>& connectedDomains, int area, int WHRatio);
	/* 提取连通域的骨架 */
	static void thinImage(cv::Mat &srcImg);
	/* 交换两个Mat */
	static void swapMat(cv::Mat &srcImg, cv::Mat &dstImg);
	/* 将图像以PNG格式保存 */
	static void save2PNG(cv::Mat &srcImg, const cv::String & fileName, int red, int green, int blue);
	/* 二值化图像。0->0,非0->255 */
	static void binaryzation(cv::Mat &srcImg);
	/* 获取图像中白点的数量 */
	static void getWhitePoints(cv::Mat &srcImg, std::vector<cv::Point>& domain);
	/* 计算宽高信息的放置位置 */
	static cv::Point calInfoPosition(int imgRows, int imgCols, int padding, const std::vector<cv::Point>& domain);

	static std::vector<std::vector<cv::Point>> detect(cv::Mat src, cv::Mat& dest);

	static void fillCrack(cv::Mat &srcImg, int width);
	static void getPoints(cv::Mat &srcImg, int i, int j, std::vector<cv::Point>& points, int squareWidth);
	static void recalculatePoints(int x, int y, std::vector<cv::Point>& points);// 暂时弃用
	static void printPoints(std::vector<cv::Point> const &points);
	static void findLines(cv::Mat & srcImg, std::vector<std::deque<cv::Point>>& _outputlines);
	static void clearArea(cv::Mat &srcImg, int i, int j, int squareWidth);

private:
	static bool findNextPoint(std::vector<cv::Point>& _neighbor_points, cv::Mat & _image, cv::Point _inpoint, int flag, cv::Point & _outpoint, int & _outflag);
	static bool findFirstPoint(cv::Mat & srcImg, cv::Point & _outputpoint);

};