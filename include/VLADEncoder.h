#include <vl/vlad.h>
#include <opencv2/core.hpp>
#include <vl/kmeans.h>
#include <vector>
#include <stdint.h>

///////////////////////////////////////////
// NOT USED
///////////////////////////////////////////

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

	void customVladEncode(float * enc,
		float const * means, vl_size dimension, vl_size numClusters,
		float const * data, vl_size numData,
		vl_uint32 const * assignments,
		int flags);
public:
	VLADEncoder(const int numCenters, const int descriptorCols);
	~VLADEncoder();
	void VLADEncoder::encode(float* enc, const Mat &descriptors);
};