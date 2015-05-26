#include "VLADEncoder.h"

#include <iostream>

void VLADEncoder::encode(float* enc, Mat &descriptors, int numCenters)
{
	vl_uint32* indexes;
	vl_uint32* assignments;
	float* data = new float[descriptors.rows * descriptors.cols];
	uint64_t numData = descriptors.rows;
	uint64_t dimensions = descriptors.cols;
	descriptors.type();

	if (descriptors.isContinuous()){
		std::copy(descriptors.datastart, descriptors.dataend, data);
	}
	else{
		for (int i = 0; i < descriptors.rows; i++)
		{
			const uint8_t* row = descriptors.ptr(i); cout << descriptors.isContinuous() << endl;
			std::copy(row, row + descriptors.cols, data + i * descriptors.cols);
		}
	}

	// create a KMeans object and run clustering to get vocabulary words (centers)
	VlKMeans* kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
	vl_kmeans_cluster(kmeans,
		data,
		dimensions,
		numData,
		numCenters);
	// find nearest cluster centers for the data that should be encoded
	indexes = new uint32_t[numData];
	assignments = new uint32_t[numData * numCenters];
	memset(assignments, 0, sizeof(vl_uint32) * numData * numCenters);
	vl_kmeans_quantize(kmeans, assignments, NULL, data, numData);
	
	// do the encoding job
	vl_vlad_encode(enc, VL_TYPE_FLOAT,
		vl_kmeans_get_centers(kmeans), dimensions, numCenters,
		data, numData,
		assignments,
		0);

	delete[] data;
	delete[] indexes;
	delete[] assignments;
	vl_kmeans_delete(kmeans);
}