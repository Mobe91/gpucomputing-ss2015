
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

int main(int argc, char** argv)
{
	VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	Mat src;

	vector<string> cifarPaths;
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

		/*printf("-- Max dist : %f \n", max_dist);
		printf("-- Min dist : %f \n", min_dist);
		printf("-- Avg dist : %f \n", avg_dist);*/

		/*cout << "A: " << endl;
		for (int i = 0; i < VLAD_CENTERS; i++)
		{
			for (int j = 0; j < AM.cols; j++)
			{
				cout << 0 + AM.at<float>(i, j) << ",";
			}
			cout << endl;
		}
		cout << "B: " << endl;
		for (int i = 0; i < VLAD_CENTERS; i++)
		{
			for (int j = 0; j < BM.cols; j++)
			{
				cout << 0 + BM.at<float>(i, j) << ",";
			}
			cout << endl;
		}*/
		//cout << "Classes: " << classA << ", " << classB<<endl;

		//cout << "norm: " << cv::norm(AM, BM, NORM_L2) << endl;

		//float resultGPU = calcDistGPU(som.d_somGrid, (float*)AM.data);
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

		/*float overall = 0;
		int mask = 0;
		for (int i = 0; i < VLAD_CENTERS; i++){
			float bestdist = std::numeric_limits<float>::max();;
			int best = 0;
			for (int j = 0; j < VLAD_CENTERS; j++){
				if ((mask & 1 << j) != 0)continue;
				float dist = 0;
				for (int k = 0; k < ORB_DESCRIPTOR_DIMENSION; k++){
					dist += fabs(AM.at<float>(k, i) - BM.at<float>(k, j));
					//if (i == 0 && j == 0) cout << fabs(AM.at<float>(k, i) - BM.at<float>(k, j)) << " from " << AM.at<float>(k, i) << " - " << BM.at<float>(k, j) << endl;
				}
				//cout << "sum " << i*VLAD_CENTERS+j << ": " << dist << endl;
				if (dist < bestdist){
					bestdist = dist;
					best = j;
				}
			}
			//cout << "match: " << i << " " << best << ": dist = " << bestdist << endl;
			overall += bestdist;
			mask |= 1 << best;
		}
		cout << "mydist: " << overall << endl;*/
			
	}

	cout << "Average difference for same classes: " << sameAVG / sameCNT << endl;
	cout << "Average difference for different classes: " << diffAVG / diffCNT << endl;

	cout << sameCNT << " same, " << diffCNT << " different" << endl;
	//VLADEncoder vladEncoder = VLADEncoder(VLAD_CENTERS, ORB_DESCRIPTOR_DIMENSION);
	while (cap.isOpened())
	{
		if (!cap.read(src))
			break;
		imshow("in", src);

		/// Detect edges
		Mat bwImage;
		Mat threshold_output;
		cv::cvtColor(src, bwImage, CV_RGB2GRAY);
		threshold(bwImage, threshold_output, 150, 255, THRESH_BINARY);
		imshow("bw", threshold_output);

		/// Find contours
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;					//TODO use hierarchy to remove unwanted objects
		findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

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
		for (int i = 0; i < contours_poly.size(); i++)
		{
			double area = contourArea(contours_poly[i]);
			if (area > 5000 && area < 300000)
			{
				// remove very small objects, and the very big ones (background)
				//draw bounding box
				boundRect[i] = boundingRect(Mat(contours_poly[i]));
				rectangle(img_box, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0);
				img_box.copyTo(img_feature);

				vector<KeyPoint> features;
				Mat descriptors;
				cv::Ptr<FeatureDetector> detector = cv::ORB::create();//(50, 1.2f, 8, 7, 0, 2, 0, 7);
				//cv::Ptr<FeatureDetector> detector = cv::cuda::ORB::create();

				//create mask  
				Mat mask = Mat::zeros(src.size(), CV_8U);
				Mat roi(mask, boundRect[i]);
				roi = Scalar(255, 255, 255);
				detector->detect(src, features, mask);				//find features
				detector->compute(src, features, descriptors);		//create feature description

				drawKeypoints(img_feature, features, img_feature, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
				cout << "desc type " << descriptors.type() << endl;
				if (descriptors.rows >= VLAD_CENTERS)
				{
					assert(descriptors.cols == ORB_DESCRIPTOR_DIMENSION);
					// allocate space for vlad encoding
					float* vlad = new float[descriptors.cols * VLAD_CENTERS];
					imshow("desc int", descriptors);
					waitKey(10);
					Mat descfloat;
					descriptors.convertTo(descfloat, CV_32FC1);
					imshow("desc float", descfloat);
					waitKey(10);
					vladEncoder.encode(vlad, descfloat);

					Mat tmp(Size(VLAD_CENTERS, ORB_DESCRIPTOR_DIMENSION), CV_32F, vlad);
					vladDesc.push_back(tmp.clone());
					for (int i = 0; i < VLAD_CENTERS; i++)
					{
						for (int j = 0; j < descriptors.cols; j++)
						{
							cout << 0 + descriptors.at<uint8_t>(i, j) << ",";
						}
						cout << endl;
					}

					cout << "----- float:" << endl;
					for (int i = 0; i < VLAD_CENTERS; i++)
					{
						for (int j = 0; j < descriptors.cols; j++)
						{
							cout << descfloat.at<float>(i, j) << ",";
						}
						cout << endl;
					}

					cout << "----- vlad:" << endl;

					for (int i = 0; i < VLAD_CENTERS; i++)
					{
						for (int j = 0; j < descriptors.cols; j++)
						{
							cout << vlad[i * descriptors.cols + j] << ",";
						}
						cout << endl;
					}

					cout << "-----" << endl;
					delete[] vlad;
				}
				//TODO call learn here
			}
		}

		if (vladDesc.size() == 2){
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
		}

		if (img_feature.size().height > 0)
		{
			imshow("Features", img_feature);
			imshow("Boxes", img_box);
		}

		waitKey(10);
	}

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