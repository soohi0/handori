#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core.hpp"

#include <queue>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;

#ifdef FINDHAND_EXPORTS
#define FINDHAND_API __declspec(dllexport)
#else
#define	FINDHAND_API __declspec(dllimport)
#endif

#define CENTER	0		// 00
#define RIGHT	1		// 01
#define LEFT	2		// 10

struct infoStruct {
	queue<int> poses;
	Point facePoint[2];
	int first;		// 손 인식 처음하는지 아닌지(얼굴 검출했는지 보기 위함)
};

struct handStruct {
	float x, y;
};

CascadeClassifier faceCascade;
int findIdx = 0;
int imgH, imgW;
Mat img;
VideoCapture cap;
int pastPose, pose;
infoStruct info;
handStruct hand;
ifstream inInfo, inQueue;
ofstream outInfo, outQueue, logFile;
string str[3] = { "first", "point1", "point2" };

extern "C" FINDHAND_API int dllStart(const char* infoFile, const char* queueFile);
extern "C" FINDHAND_API int dllUpdate();
//extern "C" FINDHAND_API int getFirstPose();

int getInfo();
void detectFace();					// 얼굴 찾기
int handColor();	// 살색으로 손 후보 결정
int findMaxContour(vector< vector<Point> > contours);	// 후보 중 손 골라내기
void fingerDetector(int idx);
void setImageSize();
int findPose(int idx);
void hideFace();
void writeLog(string str);
void writeOutInfo(string filePath, string str);
int errHandle(int err, string str); // err == 1 -> func error
void dllExit();