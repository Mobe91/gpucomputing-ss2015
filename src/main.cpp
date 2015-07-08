
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <VLADEncoder.h>
#include <CIFARImageLoader.h>
#include <SOM.h>
#include <cuda_runtime.h>
#include "Constants.h"
#include "SampleVectorGenerator.h"
#include "CalcDist.cuh"
#include <opencv2/xfeatures2d.hpp>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <opencv2\cudafeatures2d.hpp>
#include <opencv2\cudaimgproc.hpp>
#include <opencv2\cudafilters.hpp>
#include <opencv2\cudaarithm.hpp>
#include <opencv2\cudaobjdetect.hpp>
#include <opencv2\cudabgsegm.hpp>

#define DEBUG 1

const bool illuminationCorrectionEnabled = true;

using namespace cv;
using namespace std;

vector<vector<Mat>> som;

void initSOM(int w, int h, int feature_cnt, int desc_length);
void learnSOM(Mat descriptor);
void matchConvert(cv::Mat gpu_matches, std::vector<DMatch>& matches);

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
	VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
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
	/*vector<string> cifarPaths;
	cifarPaths.push_back("cifar-10-batches-bin\\data_batch_1.bin");
	SampleVectorGenerator sampleVectorGenerator(cifarPaths);

	cout << "Generating sample vectors" << endl;

	const int sampleVectorCount = 20;
	SampleVectorsHolder* sampleVectorHolder;
	sampleVectorGenerator.generateSampleVectorsFromCIFAR(&sampleVectorHolder, sampleVectorCount);

	cout << "Generated " << sampleVectorHolder->getSampleVectorCount() << " sample vectors" << endl;

	VLADEncoder vladEncoder = VLADEncoder(VLAD_CENTERS, ORB_DESCRIPTOR_DIMENSION);
	SOM som = SOM(SOM_GRID_SIZE);
	int somInitResult;
	if ((somInitResult = som.initSOM(*sampleVectorHolder)) != 0)
	{
		cerr << "SOM initialization failed" << endl;
		return somInitResult;
	}
	float sameAVG = 0, diffAVG = 0;
	int sameCNT = 0, diffCNT = 0;
		
	for (int i = 0; i < sampleVectorCount - 1; i++)
	{
		Mat AM(Size(VLAD_CENTERS, ORB_DESCRIPTOR_DIMENSION), CV_32F, (float*)sampleVectorHolder->getSampleVectors() + i*VLAD_CENTERS*ORB_DESCRIPTOR_DIMENSION);
		Mat BM(Size(VLAD_CENTERS, ORB_DESCRIPTOR_DIMENSION), CV_32F, (float*)sampleVectorHolder->getSampleVectors() + (i + 1)*VLAD_CENTERS*ORB_DESCRIPTOR_DIMENSION);
		int classA = sampleVectorHolder->getSampleClasses()[i];
		int classB = sampleVectorHolder->getSampleClasses()[i + 1];

		FlannBasedMatcher matcher;
		std::vector< DMatch > matches;
		matcher.match(AM, BM, matches);

		double max_dist = 0; 
		double min_dist = 100, avg_dist = 0;

		//-- Quick calculation of max and min distances between keypoints
		for (int i = 0; i < AM.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
			avg_dist += dist;
		}
		avg_dist /= AM.rows;

		float resultGPU = calcDistGPU2((float*)AM.data, (float*)BM.data);
		float resultNORM = cv::norm(AM, BM, NORM_L2);
		float haussDist = haussdorfDistance(AM, BM, NORM_L2);
		float haussDistTest = haussdorfDistance(AM, AM, NORM_L2);
		if (classA == classB){
			sameAVG += avg_dist;
			sameCNT++;
		}
		else{
			diffAVG += avg_dist;
			diffCNT++;
		}

		cout << "Classes: " << classA << ", " << classB << "GPU result: " << resultGPU << " L2Norm: " << resultNORM << " Hauss: " << haussDist << " CV Matching: " << avg_dist << endl;
		//cout << "--------" << endl;	
	}

	cout << "Average difference for same classes: " << sameAVG / sameCNT << endl;
	cout << "Average difference for different classes: " << diffAVG / diffCNT << endl;

	cout << sameCNT << " same, " << diffCNT << " different" << endl;*/
	//VLADEncoder vladEncoder = VLADEncoder(VLAD_CENTERS, ORB_DESCRIPTOR_DIMENSION);
	
	cv::cuda::GpuMat gpuSrc;
	cv::cuda::GpuMat gpuGrayScaleImage, gpuProcessedImage, gpuLabImage, gpuClaheResultImage, gpuForegroundImage;
	vector<cuda::GpuMat> gpuLabPlanes(3);

	cv::cuda::GpuMat keypoints_gpu, descriptors_gpu;
	cv::cuda::GpuMat gpu_mask;
	cv::cuda::GpuMat gpuMatches;

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
		/*if (!cap.read(src))
			break;*/
		gpuSrc.upload(src);

		/// Detect edges
		Mat bwImage, grayScaleImage, gpuHostImg;
		Mat threshold_output;
		
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

			// convert CPU image to grayscale
			/*cv::cvtColor(src, grayScaleImage, CV_RGB2GRAY);
			cv::GaussianBlur(grayScaleImage, bwImage, Size(17, 17), -2.0);
			imshow("Blurred", bwImage);*/
		}
		

		/*gpuBackgroundSubstractor->apply(gpuClaheResultImage, gpuProcessedImage, -1.0, stream);
		gpuProcessedImage.download(gpuHostImg, stream);
		stream.waitForCompletion();
		imshow("Background segmentation GPU", gpuHostImg);*/

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

		// CPU Canny edge detection
		/*cv::Canny(bwImage, bwImage, 10, 10 * 3);
		imshow("Canny", bwImage);*/
		
		// GPU morphology
		imgCloseFilter->apply(gpuProcessedImage, gpuProcessedImage, stream);
		gpuProcessedImage.download(gpuHostImg, stream);
		stream.waitForCompletion();
		imshow("GPU Closing", gpuHostImg);

		// CPU morphology
		/*morphologyEx(bwImage, bwImage, MORPH_DILATE, getStructuringElement(MORPH_ELLIPSE, Size(51, 51)));
		imshow("CPU Closing", bwImage);*/

		// GPU threshold
		/*cuda::threshold(gpuProcessedImage, gpuProcessedImage, 1000., 255.0, CV_8U, stream);
		gpuProcessedImage.download(gpuHostImg, stream);
		stream.waitForCompletion();
		imshow("GPU Threshold", gpuHostImg);

		// CPU threshold
		threshold(bwImage, bwImage, 100, 255, THRESH_BINARY);
		imshow("CPU Threshold", bwImage);*/
#else
		/*gpuBackgroundSubstractor->apply(gpuSrc, gpuProcessedImage, 0.1, stream);
		gpuProcessedImage.download(gpuHostImg, stream);
		stream.waitForCompletion();*/

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
		imshow("Processed", gpuHostImg);
		/// Find contours
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;					//TODO use hierarchy to remove unwanted objects
		findContours(gpuHostImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		/// Approximate contours to polygons
		vector<vector<Point> > contours_poly(contours.size());
		
		for (int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 60, true);		//reduce the contours
		}
		///vector<Mat> vladDesc;

		vector<Rect> objectBoundRects(contours.size());
		Rect tmpRect;
		std::vector<std::pair<int, cv::cuda::GpuMat>> perObjectDescriptors;
		Mat img_feature, test(src);

		// TODO: write a kernel that clusters the descriptors according to the bounding rects
		//detector->detectAndComputeAsync(gpuGrayScaleImage, gpuProcessedImage, keypoints_gpu, descriptors_gpu, false, stream);

		int objectIdx = 0;
		for (int i = 0; i < contours_poly.size(); i++)
		{
			double area = contourArea(contours_poly[i]);
			if (area > 0)
			{
				// remove very small objects, and the very big ones (background)
				
				tmpRect = boundingRect(Mat(contours_poly[i]));
				src(tmpRect).copyTo(img_feature);

				vector<KeyPoint> features;
				Mat descriptors;

				//create mask  
				Mat mask = Mat::zeros(src.size(), CV_8U);
				
				Mat roi(mask, tmpRect);
				roi = Scalar(255, 255, 255);
				//detector2->detect(src, features, mask);				//find features
				//detector2->compute(src, features, descriptors);		//create feature description

				gpu_mask.upload(mask, stream);
				detector->detectAndComputeAsync(gpuGrayScaleImage, gpu_mask, keypoints_gpu, descriptors_gpu, false, stream);
				
				stream.waitForCompletion();
				detector->convert(keypoints_gpu, features);

				// convert descriptors matrix to float - only necessary if using ORB
				//descriptors.convertTo(descriptors, CV_32FC1);

				if (descriptors_gpu.rows > 0)
				{
					objectBoundRects[objectIdx] = tmpRect;
					perObjectDescriptors.push_back(pair<int, cv::cuda::GpuMat>(objectIdx++, descriptors_gpu));
					drawKeypoints(src, features, test, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
				}
			}
		}

		//FlannBasedMatcher matcher;
		std::vector< DMatch > matches;
		
		Mat distances = Mat(perObjectDescriptors.size(), perObjectDescriptors.size(), CV_32F);

		// calculate pair-wise distances
		for (int idxOuter = 0; idxOuter < perObjectDescriptors.size(); idxOuter++)
		{
			cv::cuda::GpuMat& outer = perObjectDescriptors.at(idxOuter).second;
			for (int idxInner = 0; idxInner < perObjectDescriptors.size(); idxInner++)
			{
				if (idxInner > idxOuter){
					cv::cuda::GpuMat& inner = perObjectDescriptors.at(idxInner).second;

					//Mat matches;
					cv::cuda::Stream stream;
					matcher->matchAsync(outer, inner, gpuMatches, noArray(), stream);
					Mat hostMatches;
					gpuMatches.download(hostMatches, stream);
					
					stream.waitForCompletion();
					matchConvert(hostMatches, matches);

					//matcher.match(outer, inner, matches);

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

		const double distanceThreshold = 0.1; // TODO: experiment
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
					currentObjectMatchings.push_back(perObjectDescriptors.at(colIdx).first);
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
				objectToClusterMap.insert(std::pair<int, int>(perObjectDescriptors.at(descriptorsIdx).first, clusterIdx));
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

		imshow("Boxes", src);
		waitKey(0);
	}

		
	cap.release();
	cudaDeviceReset();

	return 0;
}

void initSOM(int w, int h, int feature_cnt, int desc_length){
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
			int distance = 0;						//TODO define metric !
			if (distance < best_distance)
			{
				best_distance = distance;
				best.x = i;
				best.y = j;
			}
		}
	}

	//TODO update neighborhood
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