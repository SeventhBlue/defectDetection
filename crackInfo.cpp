#include "crackInfo.h"

CrackInfo::CrackInfo() {}

CrackInfo::~CrackInfo() {}

CrackInfo::CrackInfo(cv::Point& position, long length, float width) :Position(position), Length(length), Width(width) {}

std::ostream& operator << (std::ostream& os, const CrackInfo& crackInfo) {
	os << "L:" << crackInfo.Length << " " << "W:";
	os.precision(3);
	os << crackInfo.Width;
	return os;
}