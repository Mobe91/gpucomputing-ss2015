#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <algorithm>
#include <math.h>
#include "CIFARImageLoader.h"
#include "SampleVectorsHolder.h"

#pragma once

using namespace std;

class SOM {
private:
	const int gridSize;
	float* d_somGrid;
	int dimensionX, dimensionY;
	int mapRadius;
	int neighborhoodRadius;
	int maxIterationNum;		//if we use constant learning rate and neighborhood size, some variables will be deleted
	int currentIterarion;
	double timeConst;
	double currentLearningRate, initialLearningRate;
	double learningDist;

public:
	vector<vector<cv::Mat>> som;

	SOM::SOM(const int gridSize);
	SOM::~SOM();

	/**
	 * Allocate a 2D grid with dimension gridSize * gridSize on the device.
	 * Each grid element constitutes a matrix with dimension ORB_DESCRIPTOR_DIMENSION * VLAD_CENTERS.
	 * Initialize the matrix elements with random values.
	 */
	int SOM::initSOM(SampleVectorsHolder &vectorsHolder);
	void SOM::learnSOM(cv::Mat descriptor);
	// TODO: add interfaces for lookup
};
