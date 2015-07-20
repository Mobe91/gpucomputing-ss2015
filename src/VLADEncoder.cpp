#include "VLADEncoder.h"
#include <iostream>

///////////////////////////////////////////
// NOT USED
///////////////////////////////////////////

#define VL_DEBUG 0

#if VL_DEBUG==1
void printMatrix(const float *data, int rows, int columns){
	for (int row = 0; row < (signed)rows; row++) {
		for (int col = 0; col < (signed)columns; col++) {
			cout << data[col*rows + row] << ",";
		}
		cout << endl;
	}
}
#endif

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

void VLADEncoder::customVladEncode(float * enc,
	float const * means, vl_size dimension, vl_size numClusters,
	float const * data, vl_size numData,
	vl_uint32 const * assignments, int flags)
{
	vl_uindex dim;
	vl_index i_cl, i_d;

	memset(enc, 0, sizeof(float) * dimension * numClusters);

#if VL_DEBUG==1
	cout << "Means" << endl;
	printMatrix(means, numClusters, dimension);
#endif

	for (i_d = 0; i_d < (signed)numData; i_d++) {
		vl_uint32 clusterIndex = assignments[i_d];
		assert(clusterIndex >= 0 && clusterIndex < numClusters);

		for (dim = 0; dim < dimension; dim++) {
			enc[clusterIndex*dimension + dim] += data[i_d  * dimension + dim] - means[clusterIndex*dimension + dim];
		}
	}

#if VL_DEBUG==1
	cout << "My unnormalized VLAD" << endl;
	printMatrix(enc, numClusters, dimension);
#endif

	// normalizations
	for (i_cl = 0; i_cl < (signed)numClusters; i_cl++) {
		double clusterMass = 0;
		if (flags & VL_VLAD_FLAG_SQUARE_ROOT) {
			for (dim = 0; dim < dimension; dim++) {
				float z = enc[i_cl*dimension + dim];
				if (z >= 0) {
					enc[i_cl*dimension + dim] = vl_sqrt_f(z);
				}
				else {
					enc[i_cl*dimension + dim] = -vl_sqrt_f(-z);
				}
			}
		}

		if (flags & VL_VLAD_FLAG_NORMALIZE_COMPONENTS) {
			float n = 0;
			dim = 0;
			for (dim = 0; dim < dimension; dim++) {
				float z = enc[i_cl*dimension + dim];
				n += z * z;
			}
			n = vl_sqrt_f(n);
			n = VL_MAX(n, 1e-12);
			for (dim = 0; dim < dimension; dim++) {
				enc[i_cl*dimension + dim] /= n;
			}
		}
	}

	if (!(flags & VL_VLAD_FLAG_UNNORMALIZED)) {
		float n = 0;
		for (dim = 0; dim < dimension * numClusters; dim++) {
			float z = enc[dim];
			n += z * z;
		}
		n = vl_sqrt_f(n);
		n = VL_MAX(n, 1e-12);
		for (dim = 0; dim < dimension * numClusters; dim++) {
			enc[dim] /= n;
		}
	}

#if VL_DEBUG==1
	cout << "My normalized VLAD" << endl;
	printMatrix(enc, numClusters, dimension);
#endif
}

void VLADEncoder::encode(float* enc, const Mat &descriptors)
{
	uint64_t numData = descriptors.rows;
	uint64_t dimensions = descriptors.cols;
	
	data->reserve(numData * descriptorCols);
	assignments->reserve(numData * numCenters);

	if (descriptors.isContinuous())
	{
		data->resize(numData * descriptorCols);
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
	//vl_kmeans_quantize(kmeans, assignments->data(), NULL, data->data(), numData);
	

	const float *means = (const float*) vl_kmeans_get_centers(kmeans);
	memcpy(enc, means, sizeof(float) * numCenters * dimensions);
	// do the encoding job
	/*customVladEncode(enc,
		(float const *) vl_kmeans_get_centers(kmeans),
		dimensions, 
		numCenters,
		(float const *) data->data(),
		numData,
		assignments->data(),
		VL_VLAD_FLAG_NORMALIZE_COMPONENTS);*/
}