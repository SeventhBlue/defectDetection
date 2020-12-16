#include "crackDetection.h"

CrackDetection::CrackDetection() {}

CrackDetection::~CrackDetection() {}

void CrackDetection::addContrast(cv::Mat& srcImg) {
	cv::Mat lookUpTable(1, 256, CV_8U);
	double temp = pow(1.1, 5);
	uchar* p = lookUpTable.data;
	for (int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>(i * temp);
	LUT(srcImg, lookUpTable, srcImg);
}

void CrackDetection::findConnectedDomain(cv::Mat & srcImg, std::vector<std::vector<cv::Point>>& connectedDomains, int area, int WHRatio) {
	cv::Mat_<uchar> tempImg = (cv::Mat_<uchar> &)srcImg;

	for (int i = 0; i < tempImg.rows; ++i) {
		uchar* row = tempImg.ptr(i);
		for (int j = 0; j < tempImg.cols; ++j) {
			if (row[j] == 255) {
				std::stack<cv::Point> connectedPoints;
				std::vector<cv::Point> domain;
				connectedPoints.push(cv::Point(j, i));
				while (!connectedPoints.empty()) {
					cv::Point currentPoint = connectedPoints.top();
					domain.push_back(currentPoint);

					int colNum = currentPoint.x;
					int rowNum = currentPoint.y;

					tempImg.ptr(rowNum)[colNum] = 0;
					connectedPoints.pop();

					if (rowNum - 1 >= 0 && colNum - 1 >= 0 && tempImg.ptr(rowNum - 1)[colNum - 1] == 255) {
						tempImg.ptr(rowNum - 1)[colNum - 1] = 0;
						connectedPoints.push(cv::Point(colNum - 1, rowNum - 1));
					}
					if (rowNum - 1 >= 0 && tempImg.ptr(rowNum - 1)[colNum] == 255) {
						tempImg.ptr(rowNum - 1)[colNum] = 0;
						connectedPoints.push(cv::Point(colNum, rowNum - 1));
					}
					if (rowNum - 1 >= 0 && colNum + 1 < tempImg.cols && tempImg.ptr(rowNum - 1)[colNum + 1] == 255) {
						tempImg.ptr(rowNum - 1)[colNum + 1] = 0;
						connectedPoints.push(cv::Point(colNum + 1, rowNum - 1));
					}
					if (colNum - 1 >= 0 && tempImg.ptr(rowNum)[colNum - 1] == 255) {
						tempImg.ptr(rowNum)[colNum - 1] = 0;
						connectedPoints.push(cv::Point(colNum - 1, rowNum));
					}
					if (colNum + 1 < tempImg.cols && tempImg.ptr(rowNum)[colNum + 1] == 255) {
						tempImg.ptr(rowNum)[colNum + 1] = 0;
						connectedPoints.push(cv::Point(colNum + 1, rowNum));
					}
					if (rowNum + 1 < tempImg.rows && colNum - 1 > 0 && tempImg.ptr(rowNum + 1)[colNum - 1] == 255) {
						tempImg.ptr(rowNum + 1)[colNum - 1] = 0;
						connectedPoints.push(cv::Point(colNum - 1, rowNum + 1));
					}
					if (rowNum + 1 < tempImg.rows && tempImg.ptr(rowNum + 1)[colNum] == 255) {
						tempImg.ptr(rowNum + 1)[colNum] = 0;
						connectedPoints.push(cv::Point(colNum, rowNum + 1));
					}
					if (rowNum + 1 < tempImg.rows && colNum + 1 < tempImg.cols && tempImg.ptr(rowNum + 1)[colNum + 1] == 255) {
						tempImg.ptr(rowNum + 1)[colNum + 1] = 0;
						connectedPoints.push(cv::Point(colNum + 1, rowNum + 1));
					}
				}
				if (domain.size() > area) {
					cv::RotatedRect rect = cv::minAreaRect(domain);
					float width = rect.size.width;
					float height = rect.size.height;
					if (width < height) {
						float temp = width;
						width = height;
						height = temp;
					}
					if (width > height * WHRatio && width > 50) {
						for (auto cit = domain.begin(); cit != domain.end(); ++cit) {
							tempImg.ptr(cit->y)[cit->x] = 250;
						}
						connectedDomains.push_back(domain);
					}
				}
			}
		}
	}

	CrackDetection::binaryzation(srcImg);
}

void CrackDetection::thinImage(cv::Mat & srcImg) {
	std::vector<cv::Point> deleteList;
	int neighbourhood[9];
	int nl = srcImg.rows;
	int nc = srcImg.cols;
	bool inOddIterations = true;
	while (true) {
		for (int j = 1; j < (nl - 1); j++) {
			uchar* data_last = srcImg.ptr<uchar>(j - 1);
			uchar* data = srcImg.ptr<uchar>(j);
			uchar* data_next = srcImg.ptr<uchar>(j + 1);
			for (int i = 1; i < (nc - 1); i++) {
				if (data[i] == 255) {
					int whitePointCount = 0;
					neighbourhood[0] = 1;
					if (data_last[i] == 255) neighbourhood[1] = 1;
					else  neighbourhood[1] = 0;
					if (data_last[i + 1] == 255) neighbourhood[2] = 1;
					else  neighbourhood[2] = 0;
					if (data[i + 1] == 255) neighbourhood[3] = 1;
					else  neighbourhood[3] = 0;
					if (data_next[i + 1] == 255) neighbourhood[4] = 1;
					else  neighbourhood[4] = 0;
					if (data_next[i] == 255) neighbourhood[5] = 1;
					else  neighbourhood[5] = 0;
					if (data_next[i - 1] == 255) neighbourhood[6] = 1;
					else  neighbourhood[6] = 0;
					if (data[i - 1] == 255) neighbourhood[7] = 1;
					else  neighbourhood[7] = 0;
					if (data_last[i - 1] == 255) neighbourhood[8] = 1;
					else  neighbourhood[8] = 0;
					for (int k = 1; k < 9; k++) {
						whitePointCount = whitePointCount + neighbourhood[k];
					}
					if ((whitePointCount >= 2) && (whitePointCount <= 6)) {
						int ap = 0;
						if ((neighbourhood[1] == 0) && (neighbourhood[2] == 1)) ap++;
						if ((neighbourhood[2] == 0) && (neighbourhood[3] == 1)) ap++;
						if ((neighbourhood[3] == 0) && (neighbourhood[4] == 1)) ap++;
						if ((neighbourhood[4] == 0) && (neighbourhood[5] == 1)) ap++;
						if ((neighbourhood[5] == 0) && (neighbourhood[6] == 1)) ap++;
						if ((neighbourhood[6] == 0) && (neighbourhood[7] == 1)) ap++;
						if ((neighbourhood[7] == 0) && (neighbourhood[8] == 1)) ap++;
						if ((neighbourhood[8] == 0) && (neighbourhood[1] == 1)) ap++;
						if (ap == 1) {
							if (inOddIterations && (neighbourhood[3] * neighbourhood[5] * neighbourhood[7] == 0)
								&& (neighbourhood[1] * neighbourhood[3] * neighbourhood[5] == 0)) {
								deleteList.push_back(cv::Point(i, j));
							}
							else if (!inOddIterations && (neighbourhood[1] * neighbourhood[5] * neighbourhood[7] == 0)
								&& (neighbourhood[1] * neighbourhood[3] * neighbourhood[7] == 0)) {
								deleteList.push_back(cv::Point(i, j));
							}
						}
					}
				}
			}
		}
		if (deleteList.size() == 0)
			break;
		for (size_t i = 0; i < deleteList.size(); i++) {
			cv::Point tem;
			tem = deleteList[i];
			uchar* data = srcImg.ptr<uchar>(tem.y);
			data[tem.x] = 0;
		}
		deleteList.clear();

		inOddIterations = !inOddIterations;
	}
}

void CrackDetection::swapMat(cv::Mat & srcImg, cv::Mat & dstImg) {
	cv::Mat tempImg = srcImg;
	srcImg = dstImg;
	dstImg = tempImg;
}

void CrackDetection::save2PNG(cv::Mat & srcImg, const cv::String & fileName, int red, int green, int blue) {
	cv::Mat dstImg(srcImg.rows, srcImg.cols, CV_8UC4, cv::Scalar(0, 0, 0, 0));
	cv::MatIterator_<uchar> srcIt = srcImg.begin<uchar>();
	cv::MatIterator_<uchar> srcEnd = srcImg.end<uchar>();
	cv::MatIterator_<cv::Vec4b> dstIt = dstImg.begin<cv::Vec4b>();
	for (; srcIt != srcEnd; ++srcIt, ++dstIt) {
		if (*srcIt > 0) {
			(*dstIt)[0] = blue;
			(*dstIt)[1] = green;
			(*dstIt)[2] = red;
			(*dstIt)[3] = 255;
		}
	}
	cv::imwrite(fileName, dstImg);
}

void CrackDetection::binaryzation(cv::Mat & srcImg) {
	cv::Mat lookUpTable(1, 256, CV_8U, cv::Scalar(255));
	lookUpTable.data[0] = 0;
	LUT(srcImg, lookUpTable, srcImg);
}

void CrackDetection::getWhitePoints(cv::Mat &srcImg, std::vector<cv::Point>& domain) {
	domain.clear();
	cv::Mat_<uchar> tempImg = (cv::Mat_<uchar> &)srcImg;
	for (int i = 0; i < tempImg.rows; i++) {
		uchar * row = tempImg.ptr<uchar>(i);
		for (int j = 0; j < tempImg.cols; ++j) {
			if (row[j] != 0)
				domain.push_back(cv::Point(j, i));
		}
	}
}

cv::Point CrackDetection::calInfoPosition(int imgRows, int imgCols, int padding, const std::vector<cv::Point>& domain) {
	long xSum = 0;
	long ySum = 0;
	for (auto it = domain.cbegin(); it != domain.cend(); ++it) {
		xSum += it->x;
		ySum += it->y;
	}
	int x = 0;
	int y = 0;
	x = (int)(xSum / domain.size());
	y = (int)(ySum / domain.size());
	if (x < padding)
		x = padding;
	if (x > imgCols - padding)
		x = imgCols - padding;
	if (y < padding)
		y = padding;
	if (y > imgRows - padding)
		y = imgRows - padding;

	return cv::Point(x, y);
}

void CrackDetection::fillCrack(cv::Mat & srcImg, int width) {
	cv::Mat_<uchar> addedContrastImg = (cv::Mat_<uchar> &)srcImg;
	for (int i = 0; i < addedContrastImg.rows; ++i) {
		uchar* row = addedContrastImg.ptr(i);
		for (int j = 0; j < addedContrastImg.cols - width; ++j) {
			int start = 0, end = 0;
			for (int delta = j; delta < j + width; ++delta) {
				if (255 == row[delta]) {
					start = delta;
					break;
				}
			}
			if (start == 0 || start == j + width - 1)
				continue;
			for (int delta = start + 1; delta < j + width; ++delta) {
				if (255 == row[delta]) {
					end = delta;
					break;
				}
			}
			if (0 == end)
				continue;
			for (; start <= end; ++start) {
				row[start] = 255;
			}
			j = end - 1;
		}
	}
}

void CrackDetection::getPoints(cv::Mat & srcImg, int i, int j, std::vector<cv::Point>& points, int squareWidth) {
	cv::Mat_<uchar> tmpImg = (cv::Mat_<uchar> &)srcImg;
	int rowLimit = (i + squareWidth) < srcImg.rows ? (i + squareWidth) : srcImg.rows;
	int colLimit = (j + squareWidth) < srcImg.cols ? (j + squareWidth) : srcImg.cols;
	for (int rowNum = i; rowNum < rowLimit; ++rowNum) {
		uchar* row = tmpImg.ptr(rowNum);
		for (int colNum = j; colNum < colLimit; ++colNum) {
			if (255 == row[colNum]) {
				points.push_back(cv::Point(colNum - j, rowNum - i));
			}
		}
	}
}

void CrackDetection::recalculatePoints(int row, int col, std::vector<cv::Point> &points) {
	for (auto begin = points.begin(); begin != points.end(); ++begin) {
		begin->x -= col;
		begin->y -= row;
	}
}

void CrackDetection::printPoints(std::vector<cv::Point> const & points) {
	std::ofstream out("points.txt");
	for (auto begin = points.cbegin(); begin != points.cend(); ++begin) {
		out << begin->x << " " << begin->y << " " << std::endl;
	}
}

void CrackDetection::clearArea(cv::Mat & srcImg, int i, int j, int squareWidth) {
	cv::Mat_<uchar> tempImg = (cv::Mat_<uchar> &)srcImg;
	int rowLimit = (i + squareWidth) < srcImg.rows ? (i + squareWidth) : srcImg.rows;
	int colLimit = (j + squareWidth) < srcImg.cols ? (i + squareWidth) : srcImg.cols;
	for (int rowNum = i; rowNum < rowLimit; ++rowNum) {
		uchar* row = tempImg.ptr(rowNum);
		for (int colNum = j; colNum < colLimit; ++colNum) {
			row[colNum] = 0;
		}
	}
}

bool CrackDetection::findNextPoint(std::vector<cv::Point> &_neighbor_points, cv::Mat &_image, cv::Point _inpoint, int flag, cv::Point& _outpoint, int &_outflag) {
	int i = flag;
	int count = 1;
	bool success = false;

	while (count <= 7) {
		cv::Point tmppoint = _inpoint + _neighbor_points[i];
		if (tmppoint.x > 0 && tmppoint.y > 0 && tmppoint.x < _image.cols&&tmppoint.y < _image.rows) {
			if (_image.at<uchar>(tmppoint) == 255) {
				_outpoint = tmppoint;
				_outflag = i;
				success = true;
				_image.at<uchar>(tmppoint) = 0;
				break;
			}
		}
		if (count % 2) {
			i += count;
			if (i > 7) {
				i -= 8;
			}
		}
		else {
			i += -count;
			if (i < 0) {
				i += 8;
			}
		}
		count++;
	}
	return success;
}

bool CrackDetection::findFirstPoint(cv::Mat &srcImg, cv::Point &_outputpoint) {
	bool success = false;
	for (int i = 0; i < srcImg.rows; i++) {
		uchar* data = srcImg.ptr<uchar>(i);
		for (int j = 0; j < srcImg.cols; j++) {
			if (data[j] == 255) {
				success = true;
				_outputpoint.x = j;
				_outputpoint.y = i;
				data[j] = 0;
				break;
			}
		}
		if (success)
			break;
	}
	return success;
}

void CrackDetection::findLines(cv::Mat & srcImg, std::vector<std::deque<cv::Point>>& _outputlines) {
	std::vector<cv::Point> neighbor_points = { cv::Point(-1,-1),cv::Point(0,-1),cv::Point(1,-1),cv::Point(1,0),
		cv::Point(1,1),cv::Point(0,1),cv::Point(-1,1),cv::Point(-1,0) };
	cv::Point first_point;
	while (findFirstPoint(srcImg, first_point)) {
		std::deque<cv::Point> line;
		line.push_back(first_point);
		//由于第一个点不一定是线段的起始位置，双向找
		cv::Point this_point = first_point;
		int this_flag = 0;
		cv::Point next_point;
		int next_flag;
		while (findNextPoint(neighbor_points, srcImg, this_point, this_flag, next_point, next_flag)) {
			line.push_back(next_point);
			this_point = next_point;
			this_flag = next_flag;
		}
		//找另一边
		this_point = first_point;
		this_flag = 0;
		//cout << "flag:" << this_flag << endl;
		while (findNextPoint(neighbor_points, srcImg, this_point, this_flag, next_point, next_flag)) {
			line.push_front(next_point);
			this_point = next_point;
			this_flag = next_flag;
		}
		if (line.size() > 10) {
			_outputlines.push_back(line);
		}
	}
}

std::vector<std::vector<cv::Point>> CrackDetection::detect(cv::Mat srcImg, cv::Mat& destImg) {
	//cv::Mat srcImgC = srcImg;

	cv::cvtColor(srcImg, destImg, cv::COLOR_BGR2GRAY, 1);

	CrackDetection::addContrast(destImg);
	CrackDetection::swapMat(srcImg, destImg);
	cv::Canny(srcImg, destImg, 50, 150);

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	dilate(destImg, destImg, kernel);
	morphologyEx(destImg, destImg, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);
	morphologyEx(destImg, destImg, cv::MORPH_CLOSE, kernel);

	std::vector<std::vector<cv::Point>> connectedDomains;
	CrackDetection::findConnectedDomain(destImg, connectedDomains, 20, 3);
	kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
	morphologyEx(destImg, destImg, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 5);

	connectedDomains.clear();
	CrackDetection::findConnectedDomain(destImg, connectedDomains, 20, 3);
	kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
	morphologyEx(destImg, destImg, cv::MORPH_OPEN, kernel);

	kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	erode(destImg, destImg, kernel);

	connectedDomains.clear();
	CrackDetection::findConnectedDomain(destImg, connectedDomains, 20, 3);

	//std::cout << "开始测量" << std::endl;
	//std::cout << "连通域数量：" << connectedDomains.size() << std::endl;
	cv::Mat lookUpTable(1, 256, CV_8U, cv::Scalar(0));
	std::vector<CrackInfo> crackInfos;
	for (auto domain_it = connectedDomains.begin(); domain_it != connectedDomains.end(); ++domain_it) {
		LUT(destImg, lookUpTable, destImg);
		for (auto point_it = domain_it->cbegin(); point_it != domain_it->cend(); ++point_it) {
			destImg.ptr<uchar>(point_it->y)[point_it->x] = 255;
		}
		double area = (double)domain_it->size();
		CrackDetection::thinImage(destImg);
		CrackDetection::getWhitePoints(destImg, *domain_it);
		long length = (long)domain_it->size();
		cv::Point position = CrackDetection::calInfoPosition(destImg.rows, destImg.cols, 50, *domain_it);
		crackInfos.push_back(CrackInfo(position, length, (float)(area / length)));
	}

	//std::cout << "开始绘制信息" << std::endl;
	//std::cout << "信息数量：" << crackInfos.size() << std::endl;

	LUT(destImg, lookUpTable, destImg);
	for (auto domain_it = connectedDomains.cbegin(); domain_it != connectedDomains.cend(); ++domain_it) {
		for (auto point_it = domain_it->cbegin(); point_it != domain_it->cend(); ++point_it) {
			destImg.ptr<uchar>(point_it->y)[point_it->x] = 255;

			//srcImgC.ptr<uchar>(point_it->y)[point_it->x * 3] = 0;
			//srcImgC.ptr<uchar>(point_it->y)[point_it->x * 3 + 1] = 0;
			//srcImgC.ptr<uchar>(point_it->y)[point_it->x * 3 + 2] = 255;
		}
	}

	//cv::imshow("drows", srcImgC);

	return connectedDomains;
}