#include "SOM.h"
#include "Constants.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda.h>
#include <time.h>

using namespace std;
using namespace cv;

SOM::SOM(const int gridSize) : gridSize(gridSize) 
{ 
	dimensionX = 10;				 //Define the dimensions of the SOM
	dimensionY = 10;
	maxIterationNum = 1000;			 //if we use constant learning rate and neighborhood size, some variables will be deleted
	initialLearningRate = 0.1;
	mapRadius = floor(max(dimensionX, dimensionY) / 2);
	timeConst = maxIterationNum / log(mapRadius);
}

SOM::~SOM()
{
	cudaFree(d_somGrid);
}
int SOM::initSOM(SampleVectorsHolder &vectorsHolder)
{
	const float* sampleVectors = vectorsHolder.getSampleVectors();
	const int sampleVectorCount = vectorsHolder.getSampleVectorCount();
	const int sampleVectorRows = vectorsHolder.getSampleVectorRows();
	const int sampleVectorCols = vectorsHolder.getSampleVectorCols();
	// som size
	int somSize = gridSize * gridSize * sampleVectorCols * sampleVectorRows;

	cout << "Initializing SOM of size " << gridSize << " x " << gridSize << " on host" << endl;

	// reserve host memory for SOM
	float* h_somGrid = new float[somSize];
	srand(time(NULL));

	for (int x = 0; x < gridSize; x++)
	{
		for (int y = 0; y < gridSize; y++)
		{
			// randomly read sample vector
			int sampleVectorIdx = rand() % sampleVectorCount;

			memcpy(
				h_somGrid + y * sampleVectorRows * sampleVectorCols * gridSize + x * sampleVectorCols * sampleVectorRows,
				sampleVectors + sampleVectorIdx * sampleVectorCols * sampleVectorRows,
				sampleVectorCols * sampleVectorRows * sizeof(float)
			);
		}
	}

	if (sampleVectorCols % 32 != 0)
	{
		cout << "WARNING: descriptor dimension " << sampleVectorCols << " is not a multiple of 32" << endl;
	}

	cout << "Allocating " << somSize * sizeof(float) << " bytes of device memory for SOM of size " << gridSize << " x " << gridSize << endl;

	cudaError_t error;

	error = cudaMalloc((void **)&d_somGrid, somSize * sizeof(float));

	if (error != ::cudaSuccess)
	{
		cerr << "Could not allocate SOM memory: " << error << endl;
		return error;
	}

	cout << "Copying intialized SOM to device" << endl;

	error = cudaMemcpy(d_somGrid, h_somGrid, somSize * sizeof(float), cudaMemcpyHostToDevice);

	if (error != ::cudaSuccess)
	{
		cerr << "Could not copy SOM grid to device: " << error << endl;
	}

	cout << "Freeing SOM host memory" << endl;
	delete[] h_somGrid;

	cout << "SOM initialization successful" << endl;
	return error;
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
	neighborhoodRadius = mapRadius * exp(-(double)currentIterarion / timeConst);

	//While the updating neigborhood neurons, reduced the searching space. Not search all the network, just search the BMU centered square. (Maybe there is more optimized way???)
	/*
	int minX = (int) max(0, (best.x - neighborhoodRadius));
	int maxX=0;
	if((best.x + neighborhoodRadius) > dimensionX){
	maxX = dimensionX;
	}else{
	maxX = (int) (best.x + neighborhoodRadius);
	}

	int minY = (int) max(0, (best.y - neighborhoodRadius));
	int maxY=0;
	if((best.y + neighborhoodRadius) > dimensionY){
	maxY = dimensionY;
	}else{
	maxY = (int) (best.y + neighborhoodRadius);
	}

	for(int i = minX; i < maxX; i++){									//replace minX, maxX
	for(int j = minY; j < maxY; j++){								//replace minY, maxY
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
	*/

	int manhattanDist; // manhattan distance between best matching unit and neuron at for loop
	for (int i = 0; som.size(); i++){
		for (int j = 0; som[i].size(); j++){
			manhattanDist = abs(best.x - i) + abs(best.y - j);
			if (manhattanDist <= neighborhoodRadius){
				currentLearningRate = initialLearningRate * exp(-(double)currentIterarion / maxIterationNum); // current learing rate is decreasing with iterations.
				learningDist = exp((-1 * pow(manhattanDist, 2)) / (2 * pow(neighborhoodRadius, 2)));	// weight of learning in the neighborhood, if the selected neuron closer to the BMU, the weight is higher.
				for (int k = 0; k < descriptor.cols; k++){
					for (int m = 0; m < descriptor.rows; m++){
						som[i][j].at<int>(k, m) = som[i][j].at<int>(k, m) + learningDist * currentLearningRate * (descriptor.at<int>(k, m) - som[i][j].at<int>(k, m)); //Update the neighborhood
					}
				}
			}
		}
	}

	currentIterarion++;

}