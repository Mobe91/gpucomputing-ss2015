#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <algorithm>
#include <math.h>

#pragma once

using namespace cv;
using namespace std;

class SOM{
private:
	int dimensionX, dimensionY;
	int mapRadius;
	int neighborhoodRadius;
	int maxIterationNum;		//if we use constant learning rate and neighborhood size, some variables will be deleted
	int currentIterarion;
	double timeConst;
	double currentLearningRate, initialLearningRate;
	double learningDist;
public:
	vector<vector<Mat>> som;

	SOM::SOM();
	void SOM::initSOM(int w, int h, int feature_cnt, int desc_length);
	void SOM::learnSOM(Mat descriptor);
};