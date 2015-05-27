#include <vl/vlad.h>
#include <opencv2/core.hpp>
#include <vl/kmeans.h>
#include <vector>
#include <stdint.h>

#pragma once

using namespace std;
using namespace cv;

class VLADEncoder
{
private:
	const int numCenters;
	const int descriptorCols;
	vector<float>* data;
	vector<vl_uint32>* assignments;
	VlKMeans* kmeans;
public:
	VLADEncoder(const int numCenters, const int descriptorCols);
	~VLADEncoder();
	void VLADEncoder::encode(float* enc, const Mat &descriptors);
};