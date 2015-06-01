#include "SOM.h"

SOM::SOM(){
	dimensionX = 10;				 //Define the dimensions of the SOM
	dimensionY = 10;
	maxIterationNum=1000;			 //if we use constant learning rate and neighborhood size, some variables will be deleted
	initialLearningRate = 0.1;
	mapRadius = floor( max(dimensionX, dimensionY) / 2 );
	timeConst =  maxIterationNum/log(mapRadius);
}

SOM::~SOM(){
}

void SOM::initSOM(int w, int h, int feature_cnt, int desc_length){
	som.resize(w);
	
	//fill the map with random values
	for (int i = 0; i < w; i++){
		som[i].resize(h);

		for (int j = 0; j < h; j++){
			Mat m = Mat(desc_length, feature_cnt, CV_8U); // TODO check orientation
			randu(m, Scalar::all(0), Scalar::all(255));

			som[i][j] = m;
		}
	}
}

void SOM::learnSOM(Mat descriptor){
	//calculate distances between object and neurons, find the best matching unit.
	Point2i best;
	double best_distance = INT_MAX;
	for (int i = 0; i < som.size(); i++){
		for (int j = 0; j < som[i].size(); j++){                        //TODO normalize descriptor matrix elements between 0 and 1??
			double distance = norm(descriptor, som[i][j], NORM_HAMMING);//TODO define metric !
			if (distance < best_distance){
				best_distance = distance;
				best.x = i;
				best.y = j;
			}
		}
	}
	//update neighborhood
	neighborhoodRadius = mapRadius * exp( -(double) currentIterarion / timeConst );

	int manhattanDist; // manhattan distance between best matching unit and neuron at for loop
	for(int i=0; som.size; i++){
		for(int j=0; som[i].size; j++){
			manhattanDist = abs(best.x - i) + abs(best.y - j);
			if(manhattanDist <= neighborhoodRadius){
				currentLearningRate = initialLearningRate * exp(-(double)currentIterarion/maxIterationNum); // current learing rate is decreasing with iterations.
				learningDist = exp( (-1 * pow(manhattanDist, 2)) / (2 * pow(neighborhoodRadius, 2)));	// weight of learning in the neighborhood, if the selected neuron closer to the BMU, the weight is higher.
				for(int k=0; k < descriptor.cols ; k++){
					for(int m=0; m < descriptor.rows ; m++){
						som[i][j].at<int>(k,m)= som[i][j].at<int>(k,m) + learningDist * currentLearningRate * (descriptor.at<int>(k,m) - som[i][j].at<int>(k,m)); //Update the neighborhood
					}				
				}
			}
		}
	}

	currentIterarion++;
	
}