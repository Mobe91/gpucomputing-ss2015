#include "SampleVectorGenerator.h"
#include "CIFARImageLoader.h"
#include "Constants.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "VLADEncoder.h"
#include <iostream>

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
	//cv::Ptr<FeatureDetector> detector = cv::ORB::create(3, 1.2f, 8, 7, 0, 2, 0, 7);

	cv::Ptr<Feature2D> detector = xfeatures2d::SIFT::create();

	vector<KeyPoint> features;
	Mat descriptors, descriptors2;
	VLADEncoder vladEncoder(VLAD_CENTERS, ORB_DESCRIPTOR_DIMENSION);

	//float min[ORB_DESCRIPTOR_DIMENSION];
	//float max[ORB_DESCRIPTOR_DIMENSION];

	// init min/max arrays
	/*for (int i = 0; i < ORB_DESCRIPTOR_DIMENSION; i++)
	{
		min[i] = std::numeric_limits<float>::max();
		max[i] = std::numeric_limits<float>::min();
	}*/
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

					/*for (int i = 0; i < VLAD_CENTERS; i++)
					{
						const uint8_t* row = descriptors.ptr(i);
						std::copy(row, row + descriptors.cols, currentSampleVector + i * descriptors.cols);
					}*/

					/*cv::norm()
					imshow("IMG1", descriptors);
					imshow("IMG2", descriptors2);
					waitKey();*/

					/*for (int i = 0; i < VLAD_CENTERS; i++)
					{
						for (int j = 0; j < ORB_DESCRIPTOR_DIMENSION; j++)
						{
							cout << currentSampleVector[i * descriptors.cols + j] << ",";
						}
						cout << endl;
					}*/

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

					/*cout << "Descriptor Means 1_1:" << endl;
					for (int i = 0; i < VLAD_CENTERS; i++)
					{
						for (int j = 0; j < ORB_DESCRIPTOR_DIMENSION; j++)
						{
							cout << currentSampleVector[i * descriptors.cols + j] << ",";
						}
						cout << endl;
					}
					cout << "-----" << endl;

					//vladEncoder.encode(currentSampleVector, descriptors);
					cout << "Descriptor Means 1_2:" << endl;
					for (int i = 0; i < VLAD_CENTERS; i++)
					{
						for (int j = 0; j < ORB_DESCRIPTOR_DIMENSION; j++)
						{
							cout << currentSampleVector[i * descriptors.cols + j] << ",";
						}
						cout << endl;
					}*/
				}
				
			}
		} while (imgPair.second != -1 && (count == 0 || sampleVectorsCount < count));
	}

	// normalize values
	/*for (int sampleVectorIdx = 0; sampleVectorIdx < sampleVectorsCount; sampleVectorIdx++)
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
	}*/

	*out = new SampleVectorsHolder(sampleVectors, sampleClass, sampleVectorsCount, VLAD_CENTERS, ORB_DESCRIPTOR_DIMENSION);
}