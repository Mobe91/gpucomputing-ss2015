#include "VLADEncoder.h"

#include <iostream>

VLADEncoder::VLADEncoder(const int numCenters, const int descriptorCols) : numCenters(numCenters), descriptorCols(descriptorCols)
{
	this->data = new vector<float>();
	this->assignments = new vector<vl_uint32>();
	this->kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
}

VLADEncoder::~VLADEncoder()
{
	delete data;
	delete assignments;
	vl_kmeans_delete(kmeans);
}

void VLADEncoder::encode(float* enc, const Mat &descriptors)
{
	uint64_t numData = descriptors.rows;
	uint64_t dimensions = descriptors.cols;
	
	if (data->size() < numData * descriptorCols)
	{
		data->resize(numData * descriptorCols);
		assignments->reserve(numData * numCenters);
	}

	if (descriptors.isContinuous())
	{
		std::copy(descriptors.datastart, descriptors.dataend, data->data());
	}
	else
	{
		for (int i = 0; i < descriptors.rows; i++)
		{
			const uint8_t* row = descriptors.ptr(i); cout << descriptors.isContinuous() << endl;
			std::copy(row, row + descriptors.cols, data->data() + i * descriptors.cols);
		}
	}

	// create a KMeans object and run clustering to get vocabulary words (centers)
	vl_kmeans_reset(kmeans);
	vl_kmeans_cluster(kmeans,
		data->data(),
		dimensions,
		numData,
		numCenters);
	// find nearest cluster centers for the data that should be encoded
	vl_kmeans_quantize(kmeans, assignments->data(), NULL, data->data(), numData);
	
	// do the encoding job
	vl_vlad_encode(enc, 
		VL_TYPE_FLOAT,
		vl_kmeans_get_centers(kmeans), 
		dimensions, 
		numCenters,
		data->data(), 
		numData,
		assignments->data(),
		0);
}