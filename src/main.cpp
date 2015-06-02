
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

using namespace cv;
using namespace std;

vector<vector<Mat>> som;

void initSOM(int w, int h, int feature_cnt, int desc_length);
void learnSOM(Mat descriptor);

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

	SampleVectorsHolder* sampleVectorHolder;
	sampleVectorGenerator.generateSampleVectorsFromCIFAR(&sampleVectorHolder);

	cout << "Generated " << sampleVectorHolder->getSampleVectorCount() << " sample vectors" << endl;

	VLADEncoder vladEncoder = VLADEncoder(VLAD_CENTERS, ORB_DESCRIPTOR_DIMENSION);
	SOM som = SOM(SOM_GRID_SIZE);
	int somInitResult;
	if ((somInitResult = som.init(*sampleVectorHolder)) != 0)
	{
		cerr << "SOM initialization failed" << endl;
		return somInitResult;
	}

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
		
		for (int i = 0; i < contours_poly.size(); i++)
		{
			double area = contourArea(contours_poly[i]);
			if (area > 5000 && area < 700000)
			{			
				// remove very small objects, and the very big ones (background)
				//draw bounding box
				boundRect[i] = boundingRect(Mat(contours_poly[i]));
				rectangle(img_box, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0);
				img_box.copyTo(img_feature);

				vector<KeyPoint> features;
				Mat descriptors;
				cv::Ptr<FeatureDetector> detector = cv::ORB::create();
				//cv::Ptr<FeatureDetector> detector = cv::cuda::ORB::create();

				//create mask  
				Mat mask = Mat::zeros(src.size(), CV_8U);
				Mat roi(mask, boundRect[i]);
				roi = Scalar(255, 255, 255);
				detector->detect(src, features, mask);				//find features
				detector->compute(src, features, descriptors);		//create feature description
				drawKeypoints(img_feature, features, img_feature, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

				if (descriptors.rows >= VLAD_CENTERS)
				{
					assert(descriptors.cols == ORB_DESCRIPTOR_DIMENSION);
					// allocate space for vlad encoding
					float* vlad = new float[descriptors.cols * VLAD_CENTERS];
					vladEncoder.encode(vlad, descriptors);
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