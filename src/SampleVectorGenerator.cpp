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

	float min[ORB_DESCRIPTOR_DIMENSION];
	float max[ORB_DESCRIPTOR_DIMENSION];

	// init min/max arrays
	for (int i = 0; i < ORB_DESCRIPTOR_DIMENSION; i++)
	{
		min[i] = std::numeric_limits<float>::max();
		max[i] = std::numeric_limits<float>::min();
	}

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
					float* currentSampleVector = sampleVectors  + sampleVectorsCount * VLAD_CENTERS * ORB_DESCRIPTOR_DIMENSION;
					vladEncoder.encode(currentSampleVector, descriptors);
					sampleVectorsCount++;

					// update min max
					for (int row = 0; row < VLAD_CENTERS; row++)
					{
						for (int col = 0; col < ORB_DESCRIPTOR_DIMENSION; col++)
						{
							min[col] = MIN(min[col], *(currentSampleVector + row * descriptors.cols + col));
							max[col] = MAX(max[col], *(currentSampleVector + row * descriptors.cols + col));
						}
					}

				}

			}
		} while (imgPair.second != -1);
	}

	// normalize values
	for (int sampleVectorIdx = 0; sampleVectorIdx < sampleVectorsCount; sampleVectorIdx++)
	{
		float* currentSampleVector = sampleVectors + sampleVectorIdx * VLAD_CENTERS * ORB_DESCRIPTOR_DIMENSION;
		for (int row = 0; row < VLAD_CENTERS; row++)
		{
			for (int col = 0; col < ORB_DESCRIPTOR_DIMENSION; col++)
			{
				float* currentVectorElement = currentSampleVector + row * ORB_DESCRIPTOR_DIMENSION + col;
				*currentVectorElement = (*currentVectorElement - min[col]) / (max[col] - min[col]);
				assert(*currentVectorElement >= 0.0f && *currentVectorElement <= 1.0f);
			}
		}
	}

	*out = new SampleVectorsHolder(sampleVectors, sampleVectorsCount, VLAD_CENTERS, ORB_DESCRIPTOR_DIMENSION);
}