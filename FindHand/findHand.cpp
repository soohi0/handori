#include "findHand.h"
#include <iostream>
#include <errno.h>
#include <time.h>

int main() {
	dllStart("./info.txt", "./queue.txt");
	return 0;
}

int dllStart(const char* infoChar, const char* queueChar) {
	string infoFile(infoChar);
	string queueFile(queueChar);

	logFile.open("./log.txt");
	logFile.close();
	
	writeLog("----dll start----\n");

	inInfo.open(infoFile);
	if (errHandle(getInfo(), "get info")) return 5;
	inInfo.close();

	/* open camera */
	cap.open(0);
	if (errHandle(!cap.isOpened(), "open camera")) return 1;

	/* set variables */
	pastPose = 0;
	pose = CENTER;

	outQueue.open(queueFile, ios_base::trunc);
	if (errHandle(outQueue.fail(), "queue file open")) return 6;

	/* start */
	if (info.first) {	// 처음 실행한거라면 얼굴 가리기 등 실행
		info.first = 0;

		outInfo.open(infoFile);
		outInfo.close();

		string str = "first/" + to_string(info.first) + "\n";
		writeOutInfo(infoFile, str);

	/* load classifier to find face */
		if (errHandle(!faceCascade.load("./haarcascade_frontalface_default.xml"),
			"load classifier cascade")) return 4;

		detectFace();
	}

	setImageSize();
	dllUpdate();

	return 0;
}

int dllUpdate() {
	int i = 0;
	while (true) {
		cap.read(img);				// 카메라로부터 이미지 읽어옴

		if (img.empty()) {			// 영상이 제대로 안됐으면 에러
			writeLog("fail:    read video err\n");
			return 1;
		}

		string str = to_string(i++) + " : success - read video\n";
		writeLog(str);

		try { //CV::exception 예외처리 handling
			if (waitKey(10) == 27) return 0;		// esc 누르면 종료

			hideFace();								// 얼굴 가리기
			fingerDetector(handColor());			// 손 위치 구별 (손의 첫번째 점을 보낸다)
		}

		catch (cv::Exception& e) {
			string err_msg = e.what();
			string str = "exception caught : " + err_msg + "\n";
			writeLog(str);
		}
	}

}

int getInfo() {
	string temp;

	string t;
	int a;

	for (int i = 0; i < sizeof(str)/sizeof(string); i++) {
		getline(inInfo, temp);

		if (temp.find(str[i]) != 0) {
			writeLog("fail:    invalid info form err\n");
			inInfo.close();
			return 1;
		}

		if (i == 0) {
			t = temp.substr(str[i].size() + 1);
			info.first = stoi(t);
		}

		else if (i == 1 || i == 2) {
			t = temp.substr(str[i].size() + 1);
			
			a = t.find('/');
			info.facePoint[i-1].x = stoi(t.substr(0, a));
			info.facePoint[i-1].y = stoi(t.substr(a + 1));
		}
	}

	inInfo.close();
	return 0;
}

int handColor() {
	Scalar handUpper(23, 255, 255);		// 손 색 upper boundary
	Scalar handLower(0, 30, 50);		// 손 색 lower boundary
	Mat imgHSV;							// RGB -> HSV한 이미지
	Size blurS(20.0, 20.0);				// 블러 사이즈
	Point blurP(-1, -1);				// 블러 포인트?
	vector< vector<Point> > contours;	// 컨투어 모음
	int handIdx;						// 컨투어 모음 중 손의 index
	Mat range;							// 살색만 남겨두고 바이너리 처리한 이미지

	cvtColor(img, imgHSV, COLOR_BGR2HSV);			// HSV로 변환한다
	inRange(imgHSV, handLower, handUpper, range);	// boundary 값에 따라 바이너리 이미지로 변환 (살색만 남겨두기)
	blur(range, range, blurS, blurP);				// 블러처리
	threshold(range, range, 180, 255, THRESH_BINARY);	// threshold

	findContours(range, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);	// 컨투어 찾기

	handIdx = findMaxContour(contours);
	// 컨투어 중 가장 큰거 찾기 -> 그게 손!

	if (handIdx == -1) {	// 크기가 너무 작으면 손이 아니다
		hand.x = -1;
		hand.y = -1;
	}
	else {
		hand.x = contours[handIdx][0].x;
		hand.y = contours[handIdx][0].y;
	}

	drawContours(img, contours, handIdx, Scalar(0, 255, 0), 2, 8);		// 손 컨투어를 원본에 덧대어 그리기

	return handIdx;
}

int findMaxContour(vector< vector<Point> > contours) {	// 후보 중 손 골라내기
	int max = 6000;	// 손이라면 크기가 6000 이상 되어야함
	int idx = -1;	// 6000을 넘는게 없을 경우 아무것도 손으로 인식 x
	int area;

	int count = contours.size();

	for (int i = 0; i < count; i++) {
		area = contourArea(contours[i]);
		if (area > max) {
			idx = i;
		}
	}

	return idx;
}

void fingerDetector(int idx) {
	findPose(idx);

	int nowPose = pose;

	if (nowPose == pastPose);
	else {
		outQueue.seekp(0, ios::end);
		outQueue << nowPose;

		pastPose = nowPose;
	}

	return;
}

int findPose(int idx) {
	if (idx == -1) {
		pose = CENTER;
		return 0;
	}

	int midptx = imgW / 2;
	int handX = hand.x;

	if (handX < midptx) pose = LEFT;
	else if (handX > midptx) pose = RIGHT;

	return 0;
}

void setImageSize() {
	cap.read(img);
	imgH = img.rows;
	imgW = img.cols;
	writeLog("success: set image size\n");
}

void hideFace() {
	Mat dst;
	flip(img, img, 1);		//좌우반전
	resize(img, dst, Size(imgW * 0.4, imgH * 0.4), 0, 0);
	imshow("display", dst);
	moveWindow("display", 20, 20);

	rectangle(img, info.facePoint[0], info.facePoint[1], Scalar(0, 0, 0), -1);// 얼굴 가리기
	rectangle(img, Point(0, 0), Point(imgW, info.facePoint[0].y), Scalar(0, 0, 0), -1);	// 얼굴 위쪽 가리기
	rectangle(img, Point(0, info.facePoint[1].y * 0.8), Point(imgW, imgH), Scalar(0, 0, 0), -1);	// 얼굴 아래쪽 가리기
}

void detectFace() {
	Mat gray;
	vector<Rect> facePos; //얼굴 위치 저장

	do {
		cap.read(img);

		cvtColor(img, gray, COLOR_BGR2GRAY);
		faceCascade.detectMultiScale(gray, facePos, 1.1, 3, 0, Size(30, 30)); //얼굴 검출

	} while (!facePos.size());

	/* facePos 크기가 0이면
	   얼굴 검출이 안 된 것이므로 검출 될때까지 계속한다 */

	info.facePoint[0] = Point(facePos[0].x, facePos[0].y * 0.7);	// 이마까지 다 가리기
	info.facePoint[1] = Point(facePos[0].x + facePos[0].width,
		facePos[0].y + (1.5 * facePos[0].height));	// 목까지 가리기

	string str1, str2;
	str1 = "point1/" + to_string(info.facePoint[0].x) + "/" + to_string(info.facePoint[0].y) + "\n";
	str2 = "point2/" + to_string(info.facePoint[1].x) + "/" + to_string(info.facePoint[1].y) + "\n";
	
	writeOutInfo("./info.txt", str1);
	writeOutInfo("./info.txt", str2);

	writeLog("success: find face\n");
}

void writeLog(string str) {
	logFile.open("./log.txt", ios::app);
	logFile << str;
	logFile.close();
}

void writeOutInfo(string filePath, string str) {
	outInfo.open(filePath, ios::app);
	outInfo << str;
	outInfo.close();
}

int errHandle(int err, string str) {
	string temp;
	if (err) {
		temp += "fail:	";
	}
	else {
		temp += "success: ";
	}

	temp += str + "\n";
	writeLog(temp);
	return err;
}