#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <opencv2/features2d.hpp>
//#include <VLADEncoder.h>
//#include <CIFARImageLoader.h>
//#include <SOM.h>
#include <cuda_runtime.h>
//#include "Constants.h"
//#include "SampleVectorGenerator.h"
//#include "CalcDist.cuh"
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <opencv2\cudafeatures2d.hpp>
#include <opencv2\cudaimgproc.hpp>
#include <opencv2\cudafilters.hpp>
#include <opencv2\cudaarithm.hpp>
#include <opencv2\cudaobjdetect.hpp>
#include <opencv2\cudabgsegm.hpp>
#include <DescriptorClusterBuilder.cuh>
#include <time.h>
#include <stdint.h>

#define DEBUG 0
#define DISPLAY_RUNTIME 1

const bool illuminationCorrectionEnabled = true;

using namespace cv;
using namespace std;

vector<vector<Mat>> som;

//void initSOM(int w, int h, int feature_cnt, int desc_length);
//void learnSOM(Mat descriptor);
void matchConvert(cv::Mat gpu_matches, std::vector<DMatch>& matches);
void putText(InputOutputArray img, Point p, ostringstream &s);

float haussdorfDistance(Mat &set1, Mat &set2, int distType)
{
	// Building distance matrix //
	Mat disMat(set1.cols, set2.cols, CV_32F);
	float *minDistances = new float[set1.rows];

	for (int row1 = 0; row1 < set1.rows; row1++)
	{
		Mat matRow1 = set1.row(row1);

		float minDist = numeric_limits<float>::max();
		for (int row2 = 0; row2 < set2.rows; row2++)
		{
			Mat matRow2 = set2.row(row2);
			float dist = norm(matRow1, matRow2, distType);
			if (dist < minDist)
			{
				minDist = dist;
			}
		}
		minDistances[row1] = minDist;
	}

	float maxDistance = numeric_limits<float>::min();
	for (int row1 = 0; row1 < set1.rows; row1++)
	{
		if (maxDistance < minDistances[row1])
		{
			maxDistance = minDistances[row1];
		}
	}
	delete[] minDistances;
	return maxDistance;
}

template<typename T> void printMatrix(const Mat& a)
{
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			cout << a.at<T>(i, j) << ",";
		}
		cout << endl;
	}
}

void f_printMatrix(const Mat& a){
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			cout << a.at<float>(i, j) << ",";
		}
		cout << endl;
	}
}

template<uint8_t> void printMatrix(Mat a);

int main(int argc, char** argv)
{
	const int viewportWidth = 1280;
	const int viewportHeight = 720;

	VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, viewportWidth);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, viewportHeight);
	Mat src;

	// define cluster colors
	const int numClusterColors = 3;
	const Scalar clusterColors[] = {
		Scalar(255, 0, 0), // red
		Scalar(0, 255, 0), // green
		Scalar(0, 0, 255) // blue
	};
	const Scalar defaultColor = Scalar(255, 255, 255);

	src = imread("vegetables.jpg");
	cv::resize(src, src, Size(1024, 768));
	
	cv::cuda::GpuMat gpuSrc;
	cv::cuda::GpuMat gpuGrayScaleImage, gpuProcessedImage, gpuLabImage, gpuClaheResultImage, gpuForegroundImage;
	vector<cuda::GpuMat> gpuLabPlanes(3);

	cv::cuda::GpuMat keypoints_gpu, descriptors_gpu;
	cv::Mat descriptors_host;
	cv::cuda::GpuMat gpu_mask;
	cv::cuda::GpuMat gpuMatches;

	cv::cuda::GpuMat gpuBoundingRects;
	Mat hostBoundingRects(10, 4, CV_16U);
	cv::cuda::GpuMat gpuClusters;

	// image filters
	cv::Ptr<cuda::Filter> gaussianFilter = cv::cuda::createGaussianFilter(CV_8U, CV_8U, Size(31, 31), -2.0);
	cv::Ptr<cuda::CannyEdgeDetector> cudaCanny = cv::cuda::createCannyEdgeDetector(5, 45);
	cv::Ptr<cuda::Filter> imgCloseFilter = cv::cuda::createMorphologyFilter(MORPH_CLOSE, CV_8U, getStructuringElement(MORPH_RECT, Size(51, 51)));
	cv::Ptr<cv::cuda::CLAHE> clahe = cv::cuda::createCLAHE();
	clahe->setClipLimit(4);

	cv::Ptr<cuda::BackgroundSubtractorMOG2> gpuBackgroundSubstractor = cv::cuda::createBackgroundSubtractorMOG2();

	// feature extractor
	cv::Ptr<cv::cuda::ORB> detector = cv::cuda::ORB::create();

	// feature matcher
	cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);

	// CUDA streams
	cv::cuda::Stream stream;
	
	while (cap.isOpened())
	{
		clock_t frameStart = clock();

		//if (!cap.read(src))
		//	break;
		gpuSrc.upload(src);

		/// Detect edges
		Mat bwImage, grayScaleImage, gpuHostImg;
		Mat threshold_output;

		clock_t imgProcessingStart = clock();
#if DEBUG==1
		// clahe illumination correction
		if (illuminationCorrectionEnabled)
		{
			cv::cuda::cvtColor(gpuSrc, gpuLabImage, CV_BGR2Lab, 0, stream);
			cuda::split(gpuLabImage, gpuLabPlanes, stream);
			clahe->apply(gpuLabPlanes[0], gpuLabPlanes[0], stream);
			cuda::merge(gpuLabPlanes, gpuLabImage, stream);
			cv::cuda::cvtColor(gpuLabImage, gpuClaheResultImage, CV_Lab2BGR, 0, stream);
			gpuClaheResultImage.download(gpuHostImg, stream);
			stream.waitForCompletion();
			imshow("Clahe GPU", gpuHostImg);

			// convert GPU image to grayscale
			cv::cuda::cvtColor(gpuClaheResultImage, gpuGrayScaleImage, CV_BGR2GRAY, 0, stream);
			gpuGrayScaleImage.download(gpuHostImg, stream);
			stream.waitForCompletion();
			imshow("GPU grayscale", gpuHostImg);
		}
		else
		{
			// convert GPU image to grayscale
			cv::cuda::cvtColor(gpuSrc, gpuGrayScaleImage, CV_BGR2GRAY, 0, stream);
			gpuGrayScaleImage.download(gpuHostImg, stream);
			stream.waitForCompletion();
			imshow("GPU grayscale", gpuHostImg);
		}

		// blur GPU image
		gaussianFilter->apply(gpuGrayScaleImage, gpuProcessedImage, stream);
		gpuProcessedImage.download(gpuHostImg, stream);
		stream.waitForCompletion();
		imshow("Blurred GPU", gpuHostImg);

		// GPU Canny edge detection
		cudaCanny->detect(gpuProcessedImage, gpuProcessedImage, stream);
		gpuProcessedImage.download(gpuHostImg, stream);
		stream.waitForCompletion();
		imshow("Canny GPU", gpuHostImg);
		
		// GPU morphology
		imgCloseFilter->apply(gpuProcessedImage, gpuProcessedImage, stream);
		gpuProcessedImage.download(gpuHostImg, stream);
		stream.waitForCompletion();
		imshow("GPU Closing", gpuHostImg);
#else
		if (illuminationCorrectionEnabled)
		{
			cv::cuda::cvtColor(gpuSrc, gpuLabImage, CV_BGR2Lab, 0, stream);
			cuda::split(gpuLabImage, gpuLabPlanes, stream);
			clahe->apply(gpuLabPlanes[0], gpuLabPlanes[0], stream);
			cuda::merge(gpuLabPlanes, gpuLabImage, stream);
			// convert GPU image to grayscale
			cv::cuda::cvtColor(gpuLabImage, gpuClaheResultImage, CV_Lab2BGR, 0, stream);
			cv::cuda::cvtColor(gpuClaheResultImage, gpuGrayScaleImage, CV_BGR2GRAY, 0, stream);
		}
		else
		{
			// convert GPU image to grayscale
			cv::cuda::cvtColor(gpuSrc, gpuGrayScaleImage, CV_BGR2GRAY, 0, stream);
		}
		
		gaussianFilter->apply(gpuGrayScaleImage, gpuProcessedImage, stream);
		cudaCanny->detect(gpuProcessedImage, gpuProcessedImage, stream);
		imgCloseFilter->apply(gpuProcessedImage, gpuProcessedImage, stream);
		gpuProcessedImage.download(gpuHostImg, stream);
		stream.waitForCompletion();
		
#endif
		clock_t imgProcessingEnd = clock();
		imshow("Processed", gpuHostImg);
		
		clock_t objectDetectionStart = clock();
		// Find contours
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(gpuHostImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		// Approximate contours to polygons
		vector<vector<Point> > contours_poly(contours.size());
		
		for (int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 60, true);		//reduce the contours
		}

		hostBoundingRects.resize(contours_poly.size());

		clock_t objectDetectionEnd = clock();

		clock_t featureExtractionStart = clock();

		vector<Rect> objectBoundRects(contours.size());
		Rect tmpRect;
		std::vector<std::pair<int, cv::Mat>> hostPerObjectDescriptors;
		std::vector<std::pair<int, cv::cuda::GpuMat>> devicePerObjectDescriptors;
		Mat img_feature;

		detector->detectAndComputeAsync(gpuGrayScaleImage, gpuProcessedImage, keypoints_gpu, descriptors_gpu, false, stream);

		int objectIdx = 0;
		for (int i = 0; i < contours_poly.size(); i++)
		{
			double area = contourArea(contours_poly[i]);
			if (area > 2500)
			{
				// remove very small objects, and the very big ones (background)
				tmpRect = boundingRect(Mat(contours_poly[i]));
				
				hostBoundingRects.ptr<short>(objectIdx)[0] = tmpRect.tl().x;
				hostBoundingRects.ptr<short>(objectIdx)[1] = tmpRect.tl().y;
				hostBoundingRects.ptr<short>(objectIdx)[2] = tmpRect.br().x;
				hostBoundingRects.ptr<short>(objectIdx)[3] = tmpRect.br().y;

				objectBoundRects[objectIdx++] = tmpRect;
			}
		}

		clock_t featureExtractionEnd = clock();

		clock_t descriptorClusteringStart = 0;
		clock_t descriptorClusteringEnd = 0;
		clock_t matchingStart = 0;
		clock_t matchingEnd = 0;
		clock_t objectClusteringStart = 0;
		clock_t objectClusteringEnd = 0;

		// only perform clustering if at least one object was detected
		if (objectIdx > 0)
		{
			descriptorClusteringStart = clock();

			// get the object bounding rectangles (hostBoundingRects was allocated with a pessimistic size)
			Mat activeRects = hostBoundingRects.rowRange(cv::Range(0, objectIdx));
			gpuBoundingRects.upload(activeRects, stream);

			// get features to draw
			vector<KeyPoint> features;
			stream.waitForCompletion();
			detector->convert(keypoints_gpu, features);
			drawKeypoints(src, features, src, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

			// create mats on host and device that will contain the clustered descriptors
			for (int i = 0; i < objectIdx; i++)
			{
				hostPerObjectDescriptors.push_back(pair<int, cv::Mat>(i, Mat(descriptors_gpu.rows, descriptors_gpu.cols, descriptors_gpu.type())));
				devicePerObjectDescriptors.push_back(pair<int, cv::cuda::GpuMat>(i, cv::cuda::GpuMat(descriptors_gpu.rows, descriptors_gpu.cols, descriptors_gpu.type())));
			}

			descriptors_gpu.download(descriptors_host, stream);
			// invoke custom kernel for descriptor clustering
			clusterORBDescriptors(gpuBoundingRects, keypoints_gpu, descriptors_host, gpuClusters, hostPerObjectDescriptors, stream);

			//FlannBasedMatcher matcher;
			std::vector< DMatch > matches;
		
			// upload clustered descriptors to GPU
			for (int objectIdx = 0; objectIdx < hostPerObjectDescriptors.size(); objectIdx++)
			{
				devicePerObjectDescriptors.at(objectIdx).second.upload(hostPerObjectDescriptors.at(objectIdx).second, stream);
			}

			descriptorClusteringEnd = clock();
			matchingStart = clock();

			Mat distances = Mat(hostPerObjectDescriptors.size(), hostPerObjectDescriptors.size(), CV_32F);

			// calculate pair-wise distances
			for (int idxOuter = 0; idxOuter < hostPerObjectDescriptors.size(); idxOuter++)
			{
				cv::cuda::GpuMat& outer = devicePerObjectDescriptors.at(idxOuter).second;
				for (int idxInner = 0; idxInner < hostPerObjectDescriptors.size(); idxInner++)
				{
					if (idxInner > idxOuter){
						cv::cuda::GpuMat& inner = devicePerObjectDescriptors.at(idxInner).second;

						//Mat matches;
						matcher->matchAsync(outer, inner, gpuMatches, noArray(), stream);
						Mat hostMatches;
						gpuMatches.download(hostMatches, stream);
					
						stream.waitForCompletion();
						matchConvert(hostMatches, matches);
						
						float avgDist = 0.0;
						float maxDist = numeric_limits<float>::min();
						for (int i = 0; i < matches.size(); i++)
						{
							float dist = matches.at(i).distance;
							avgDist += dist;
							if (dist > maxDist)
							{
								maxDist = dist;
							}
						}
						avgDist /= matches.size();
						distances.at<float>(idxOuter, idxInner) = avgDist / maxDist;
					}
					else if (idxInner < idxOuter)
					{
						distances.at<float>(idxOuter, idxInner) = distances.at<float>(idxInner, idxOuter);
					}
					else
					{
						distances.at<float>(idxOuter, idxInner) = numeric_limits<float>::max();
					}
				}
			}

			matchingEnd = clock();
			objectClusteringStart = clock();

			const double distanceThreshold = 0.13; // TODO: experiment
			std::vector<std::vector<int>> objectMatchings;
			// for each object determines the index of the most similar different objects 

#if DEBUG == 1
			// print distance matrix
			printMatrix<float>(distances);
#endif
			for (int rowIdx = 0; rowIdx < distances.rows; rowIdx++)
			{
				std::vector<int> currentObjectMatchings;
				for (int colIdx = 0; colIdx < distances.cols; colIdx++)
				{
					if (distances.at<float>(rowIdx, colIdx) < distanceThreshold)
					{
						currentObjectMatchings.push_back(hostPerObjectDescriptors.at(colIdx).first);
					}
				}
				objectMatchings.push_back(currentObjectMatchings);
			}

#if DEBUG == 1
			// Output object similarities
			for (int objectIdx = 0; objectIdx < objectMatchings.size(); objectIdx++)
			{
				for (int j = 0; j < objectMatchings.at(objectIdx).size(); j++)
				{
					cout << objectIdx << ": " << objectMatchings.at(objectIdx).at(j) << endl;
				}
			}
#endif

			int clusterCount = 0;
			std::unordered_map<int, std::unordered_set<int>> clusterToObjectMap;
			std::unordered_map<int, int> objectToClusterMap;
			for (int descriptorsIdx = 0; descriptorsIdx < distances.rows; descriptorsIdx++)
			{
				// check if this object has no cluster assigned so far
				unordered_map<int, int>::iterator existingClusterIt = objectToClusterMap.find(descriptorsIdx);
				if (existingClusterIt == objectToClusterMap.end())
				{
					// create new cluster
					std::vector<int> currentObjectMatchings = objectMatchings.at(descriptorsIdx);
					std::unordered_set<int> clusterObjects(currentObjectMatchings.begin(), currentObjectMatchings.end());
					clusterObjects.insert(descriptorsIdx);
					clusterToObjectMap.insert(std::pair<int, unordered_set<int>>(clusterCount, clusterObjects));

					for (std::unordered_set<int>::iterator clusterObjectsIter = clusterObjects.begin(); clusterObjectsIter != clusterObjects.end(); clusterObjectsIter++)
					{
						objectToClusterMap.insert(std::pair<int, int>(*clusterObjectsIter, clusterCount));
					}
					clusterCount++;
				}
				else
				{
					// populate existing clusters with similar objects of this map
					std::vector<int> currentObjectMatchings = objectMatchings.at(descriptorsIdx);
					int clusterIdx = existingClusterIt->second;
					clusterToObjectMap.at(clusterIdx).insert(currentObjectMatchings.begin(), currentObjectMatchings.end());
					// no need to insert descriptorsIdx
					objectToClusterMap.insert(std::pair<int, int>(hostPerObjectDescriptors.at(descriptorsIdx).first, clusterIdx));
				}
			}

			// draw clusters
			for (std::unordered_map<int, int>::iterator it = objectToClusterMap.begin(); it != objectToClusterMap.end(); it++)
			{
				//draw bounding box
				const Scalar* clusterColor;
				if (it->second < numClusterColors)
				{
					clusterColor = clusterColors + it->second;
				}
				else
				{
					clusterColor = &defaultColor;
				}
				Rect& objectBoundRect = objectBoundRects[it->first];
				rectangle(src, objectBoundRect.tl(), objectBoundRect.br(), *clusterColor, 2, 8, 0);
			}

			objectClusteringEnd = clock();

		}

		clock_t frameEnd = clock();

#if DISPLAY_RUNTIME == 1
		// render runtime measurements		
		double imgProcessingTime_ms = (double(imgProcessingEnd - imgProcessingStart) / CLOCKS_PER_SEC) * 1000.0;
		double objectDetectionTime_ms = (double(objectDetectionEnd - objectDetectionStart) / CLOCKS_PER_SEC) * 1000.0;
		double featureExtractionTime_ms = (double(featureExtractionEnd - featureExtractionStart) / CLOCKS_PER_SEC) * 1000.0;
		double descriptorClusteringTime_ms = (double(descriptorClusteringEnd - descriptorClusteringStart) / CLOCKS_PER_SEC) * 1000.0;
		double matchingTime_ms = (double(matchingEnd - matchingStart) / CLOCKS_PER_SEC) * 1000.0;
		double objectClusteringTime_ms = (double(objectClusteringEnd - objectClusteringStart) / CLOCKS_PER_SEC) * 1000.0;


		double frameTime_ms = (double(frameEnd- frameStart) / CLOCKS_PER_SEC) * 1000.0;

		ostringstream stringStream;
		stringStream << "Img processing time: " << imgProcessingTime_ms << " ms";
		putText(src, Point(viewportWidth - 600, viewportHeight - 200), stringStream);
		stringStream.str("");
		stringStream.clear();

		stringStream << "Object detection time: " << objectDetectionTime_ms << " ms";
		putText(src, Point(viewportWidth - 600, viewportHeight - 180), stringStream);
		stringStream.str("");
		stringStream.clear();

		stringStream << "Feature extraction time: " << featureExtractionTime_ms << " ms";
		putText(src, Point(viewportWidth - 600, viewportHeight - 160), stringStream);
		stringStream.str("");
		stringStream.clear();

		stringStream << "Descriptor clustering time: " << descriptorClusteringTime_ms << " ms";
		putText(src, Point(viewportWidth - 600, viewportHeight - 140), stringStream);
		stringStream.str("");
		stringStream.clear();

		stringStream << "Matching time: " << matchingTime_ms << " ms";
		putText(src, Point(viewportWidth - 600, viewportHeight - 120), stringStream);
		stringStream.str("");
		stringStream.clear();

		stringStream << "Object clustering time: " << objectClusteringTime_ms << " ms";
		putText(src, Point(viewportWidth - 600, viewportHeight - 100), stringStream);
		stringStream.str("");
		stringStream.clear();

		stringStream << "Frame time: " << frameTime_ms << " ms";
		putText(src, Point(viewportWidth - 600, viewportHeight - 80), stringStream);
		stringStream.str("");
		stringStream.clear();
		
#endif
		
		imshow("Boxes", src);
		waitKey(0);
	}

		
	cap.release();
	cudaDeviceReset();

	return 0;
}

void putText(InputOutputArray img, Point p, ostringstream &s)
{
	putText(img, s.str(), p, FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(250, 250, 250));
}

void matchConvert(cv::Mat gpu_matches, std::vector<DMatch>& matches)
{
	if (gpu_matches.empty())
	{
		matches.clear();
		return;
	}

	CV_Assert((gpu_matches.type() == CV_32SC1) && (gpu_matches.rows == 2 || gpu_matches.rows == 3));

	const int nQuery = gpu_matches.cols;

	matches.clear();
	matches.reserve(nQuery);

	const int* trainIdxPtr = NULL;
	const int* imgIdxPtr = NULL;
	const float* distancePtr = NULL;

	if (gpu_matches.rows == 2)
	{
		trainIdxPtr = gpu_matches.ptr<int>(0);
		distancePtr = gpu_matches.ptr<float>(1);
	}
	else
	{
		trainIdxPtr = gpu_matches.ptr<int>(0);
		imgIdxPtr = gpu_matches.ptr<int>(1);
		distancePtr = gpu_matches.ptr<float>(2);
	}

	for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
	{
		const int trainIdx = trainIdxPtr[queryIdx];
		if (trainIdx == -1)
			continue;

		const int imgIdx = imgIdxPtr ? imgIdxPtr[queryIdx] : 0;
		const float distance = distancePtr[queryIdx];

		DMatch m(queryIdx, trainIdx, imgIdx, distance);

		matches.push_back(m);
	}
}

/*void initSOM(int w, int h, int feature_cnt, int desc_length){
	som.resize(w);

	//fill the map with random values
	for (int i = 0; i < w; i++)
	{
		som[i].resize(h);

		for (int j = 0; j < h; j++)
		{
			Mat m = Mat(desc_length, feature_cnt, CV_8U); // TODO check orientation
			randu(m, Scalar::all(0), Scalar::all(255));

			som[i][j] = m;
		}
	}
}

void learnSOM(Mat descriptor)
{
	Point2i best;
	int best_distance = INT_MAX;
	for (int i = 0; i < som.size(); i++)
	{
		for (int j = 0; j < som[i].size(); j++)
		{
			int distance = 0;
			if (distance < best_distance)
			{
				best_distance = distance;
				best.x = i;
				best.y = j;
			}
		}
	}
}*/	