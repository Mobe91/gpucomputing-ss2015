#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <algorithm>
#include <math.h>
#include "CIFARImageLoader.h"
#include "SampleVectorsHolder.h"

///////////////////////////////////////////
// NOT USED
///////////////////////////////////////////

#pragma once

using namespace std;

class SOM {
private:
	
public:
	const int gridSize;
	float* d_somGrid;
	int dimensionX, dimensionY;
	int mapRadius;
	float neighborhoodRadius;
	int maxIterationNum;		//if we use constant learning rate and neighborhood size, some variables will be deleted
	int currentIterarion;
	float timeConst;
	float currentLearningRate, initialLearningRate;
	float learningDist;
	
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
