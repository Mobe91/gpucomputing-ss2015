
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

using namespace cv;
using namespace std;

vector<vector<Mat>> som;

void initSOM(int w, int h, int feature_cnt, int desc_length);
void learnSOM(Mat descriptor);

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
	//while (cap.isOpened())
	//{
		/*if (!cap.read(src))
			break;*/

		imshow("in", src);

		/// Detect edges
		Mat bwImage;
		Mat threshold_output;
		cv::cvtColor(src, bwImage, CV_RGB2GRAY);
		cv::blur(bwImage, bwImage, Size(11, 11));
		//imshow("Blurred", bwImage);
		cv::Canny(bwImage, bwImage, 10, 10 * 3, 3);
		//imshow("Canny", bwImage);
		//cv::adaptiveThreshold(bwImage, threshold_output, 255.0, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
		morphologyEx(bwImage, bwImage, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(51,51)));
		//morphologyEx(bwImage, bwImage, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		//imshow("Closing", bwImage);
		threshold(bwImage, threshold_output, 100, 255, THRESH_BINARY);
		//imshow("bw", threshold_output);

		/// Find contours
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;					//TODO use hierarchy to remove unwanted objects
		findContours(threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		/// Approximate contours to polygons
		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 60, true);		//reduce the contours
		}

		Mat img_feature, img_box;

		src.copyTo(img_box);
		vector<Mat> vladDesc;
		std::vector<std::pair<int, Mat>> perObjectDescriptors;
		for (int i = 0; i < contours_poly.size(); i++)
		{
			double area = contourArea(contours_poly[i]);
			if (area > 0)
			{
				// remove very small objects, and the very big ones (background)
				boundRect[i] = boundingRect(Mat(contours_poly[i]));
				img_box(boundRect[i]).copyTo(img_feature);
				vector<KeyPoint> features;
				Mat descriptors;
				cv::Ptr<FeatureDetector> detector = ORB::create();
				//cv::Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();
				//cv::Ptr<FeatureDetector> detector = cv::cuda::ORB::create();

				//create mask  
				Mat mask = Mat::zeros(src.size(), CV_8U);
				Mat roi(mask, boundRect[i]);
				roi = Scalar(255, 255, 255);
				detector->detect(src, features, mask);				//find features
				detector->compute(src, features, descriptors);		//create feature description

				// convert descriptors matrix to float - only necessary if using ORB
				descriptors.convertTo(descriptors, CV_32FC1);

				if (descriptors.rows > 0)
				{
					perObjectDescriptors.push_back(pair<int, Mat>(i, descriptors));
					drawKeypoints(img_feature, features, img_feature, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
				}
				//TODO call learn here
			}
		}

		FlannBasedMatcher matcher;
		std::vector< DMatch > matches;
		
		
		Mat distances = Mat(perObjectDescriptors.size(), perObjectDescriptors.size(), CV_32F);
		// calculate pair-wise distances
		for (int idxOuter = 0; idxOuter < perObjectDescriptors.size(); idxOuter++)
		{
			Mat& outer = perObjectDescriptors.at(idxOuter).second;
			for (int idxInner = 0; idxInner < perObjectDescriptors.size(); idxInner++)
			{
				if (idxInner > idxOuter){
					Mat& inner = perObjectDescriptors.at(idxInner).second;
					matcher.match(outer, inner, matches);

					float avgDist = 0.0;
					for (int i = 0; i < matches.size(); i++)
					{
						float dist = matches[i].distance;
						avgDist += dist;
					}
					avgDist /= matches.size();
					distances.at<float>(idxOuter, idxInner) = avgDist;
				}
				else
				{
					distances.at<float>(idxOuter, idxInner) = numeric_limits<float>::max();
				}
			}
		}

		const double distanceThreshold = 10; // TODO: experiment
		std::vector<std::vector<int>> objectMatchings(perObjectDescriptors.size());
		// for each object determines the index of the most similar different objects 
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
				clusterToObjectMap.insert(std::pair<int, unordered_set<int>>(clusterCount, std::unordered_set<int>(currentObjectMatchings.begin(), currentObjectMatchings.end())));
				objectToClusterMap.insert(std::pair<int, int>(perObjectDescriptors.at(descriptorsIdx).first, clusterCount++));
			}
			else
			{
				// populate existing clusters with similar objects of this map
				std::vector<int> currentObjectMatchings = objectMatchings.at(descriptorsIdx);
				int clusterIdx = existingClusterIt->second;
				clusterToObjectMap.at(clusterIdx).insert(currentObjectMatchings.begin(), currentObjectMatchings.end());
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
			rectangle(img_box, boundRect[it->first].tl(), boundRect[it->first].br(), *clusterColor, 2, 8, 0);
		}
		


		/*if (vladDesc.size() == 2){
			cout << "norm: " << cv::norm(vladDesc[0], vladDesc[1], NORM_L2) << endl;
			float overall = 0;
			int mask = 0;
			for (int i = 0; i < VLAD_CENTERS; i++){
				float bestdist = std::numeric_limits<float>::max();;
				int best = 0;
				for (int j = 0; j < VLAD_CENTERS; j++){
					if ((mask & 1 << j) != 0)continue;
					float dist = 0;
					for (int k = 0; k < ORB_DESCRIPTOR_DIMENSION; k++){
						dist += fabs(vladDesc[0].at<float>(k, i) - vladDesc[1].at<float>(k, j));
					}
					if (dist < bestdist){
						bestdist = dist;
						best = j;
					}
				}
				overall += bestdist;
				mask |= 1 << best;
			}
			cout << "mydist: " << overall << endl;
		}*/

		if (img_feature.size().height > 0)
		{
			imshow("Features", img_feature);
			imshow("Boxes", img_box);
		}

		waitKey();
	//}

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