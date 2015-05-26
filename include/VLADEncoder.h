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
public:
	void VLADEncoder::encode(float* enc, Mat &descriptors, int numCenters);
};