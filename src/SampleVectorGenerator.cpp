#include "SampleVectorGenerator.h"
#include "CIFARImageLoader.h"
#include "Constants.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "VLADEncoder.h"
#include <iostream>

///////////////////////////////////////////
// NOT USED
///////////////////////////////////////////


using namespace cv;

SampleVectorGenerator::SampleVectorGenerator(vector<string> rawCIFARFilePaths) : loaders() {
	loaders.reserve(rawCIFARFilePaths.size());
	totalPictureCount = 0;
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

void SampleVectorGenerator::generateSampleVectorsFromCIFAR(SampleVectorsHolder** out, const int count)
{
	// allocate memory for sample vectors
	float* sampleVectors = new float[totalPictureCount * ORB_DESCRIPTOR_DIMENSION * VLAD_CENTERS];
	int *sampleClass = new int[totalPictureCount];
	int sampleVectorsCount = 0;
	pair<Mat, int> imgPair;

	cv::Ptr<Feature2D> detector = xfeatures2d::SIFT::create();

	vector<KeyPoint> features;
	Mat descriptors, descriptors2;
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

				descriptors2 = 0;
				detector->detect(imgPair.first, features);				//find features
				detector->compute(imgPair.first, features, descriptors2);		//create feature description

				if (descriptors.rows >= VLAD_CENTERS)
				{
					float* currentSampleVector = sampleVectors + sampleVectorsCount * VLAD_CENTERS * ORB_DESCRIPTOR_DIMENSION;

					Mat floatDescriptors;
					descriptors.convertTo(floatDescriptors, CV_32F);
					Mat centers, labels;
					cv::kmeans(floatDescriptors, VLAD_CENTERS, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, 0, centers);
					for (int i = 0; i < centers.rows; i++)
					{
						const uint8_t* row = descriptors.ptr(i);
						std::copy(row, row + descriptors.cols, currentSampleVector + i * descriptors.cols);
					}

					//vladEncoder.encode(currentSampleVector, descriptors);
					sampleClass[sampleVectorsCount] = imgPair.second;
					sampleVectorsCount++;
				}
				
			}
		} while (imgPair.second != -1 && (count == 0 || sampleVectorsCount < count));
	}

	*out = new SampleVectorsHolder(sampleVectors, sampleClass, sampleVectorsCount, VLAD_CENTERS, ORB_DESCRIPTOR_DIMENSION);
}