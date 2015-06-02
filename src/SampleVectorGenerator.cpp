#include "SampleVectorGenerator.h"
#include "CIFARImageLoader.h"
#include "Constants.h"
#include <opencv2/features2d.hpp>
#include "VLADEncoder.h"

using namespace cv;

SampleVectorGenerator::SampleVectorGenerator(vector<string> rawCIFARFilePaths) : loaders() {
	loaders.reserve(rawCIFARFilePaths.size());
	for (auto &rawCIFAR : rawCIFARFilePaths)
	{
		CIFARImageLoader* imgLoader = new CIFARImageLoader(rawCIFAR);
		loaders.push_back(imgLoader);
		totalPictureCount += imgLoader->getPictureCount();
	}
}

SampleVectorGenerator::~SampleVectorGenerator() {
	// close CIFRA loaders
	for (auto &loader : loaders)
	{
		delete loader;
	}
}

void SampleVectorGenerator::generateSampleVectorsFromCIFAR(SampleVectorsHolder** out)
{
	// allocate memory for sample vectors
	float* sampleVectors = new float[totalPictureCount * ORB_DESCRIPTOR_DIMENSION * VLAD_CENTERS];
	int sampleVectorsCount = 0;
	pair<Mat, int> imgPair;
	cv::Ptr<FeatureDetector> detector = cv::ORB::create(50, 1.2f, 8, 7, 0, 2, 0, 7);
	vector<KeyPoint> features;
	Mat descriptors;
	VLADEncoder vladEncoder(VLAD_CENTERS, ORB_DESCRIPTOR_DIMENSION);

	for (auto &loader : loaders)
	{
		do {
			loader->getNextImage(imgPair);
			if (imgPair.second != -1)
			{
				features.clear();
				descriptors = 0;
				detector->detect(imgPair.first, features);				//find features
				detector->compute(imgPair.first, features, descriptors);		//create feature description

				if (descriptors.rows >= VLAD_CENTERS)
				{
					vladEncoder.encode(sampleVectors + sampleVectorsCount * VLAD_CENTERS * ORB_DESCRIPTOR_DIMENSION, descriptors);
					sampleVectorsCount++;
				}

			}
		} while (imgPair.second != -1);
	}

	*out = new SampleVectorsHolder(sampleVectors, sampleVectorsCount, VLAD_CENTERS, ORB_DESCRIPTOR_DIMENSION);
}