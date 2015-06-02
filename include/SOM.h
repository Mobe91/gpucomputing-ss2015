#include "CIFARImageLoader.h"
#include "SampleVectorsHolder.h"

#pragma once

class SOM {
private:
	const int gridSize;
	float* d_somGrid;

	void SOM::getNextImageCycling(CIFARImageLoader &imgLoader, pair<cv::Mat, int> &out);
public:
	SOM(const int gridSize);
	~SOM();

	/**
	 * Allocate a 2D grid with dimension gridSize * gridSize on the device.
	 * Each grid element constitutes a matrix with dimension ORB_DESCRIPTOR_DIMENSION * VLAD_CENTERS.
	 * Initialize the matrix elements with random values.
	 */
	int init(SampleVectorsHolder &vectorsHolder);
	// TODO: add interfaces for training and lookup
};