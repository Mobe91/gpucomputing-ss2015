#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda.h>
#include <device_functions.h>
#include "UpdateSOM.cuh"
#include "Constants.h"
#include <iostream>
#include "SOM.h"
#include <math.h> 

///////////////////////////////////////////
// NOT USED
///////////////////////////////////////////

__global__ void updateSOMKernel(float *d_somGrid, float *input, int indexOfBMU, float neighborhoodRadius, float currentLearningRate){

	int blockXPosition = blockIdx.x / SOM_GRID_SIZE;
	int blockYPosition = blockIdx.x % SOM_GRID_SIZE;
	int BMUXPosition =	indexOfBMU / SOM_GRID_SIZE;
	int BMUYPosition =	indexOfBMU % SOM_GRID_SIZE;
	float manhattanDist = abs(blockXPosition - BMUXPosition) + abs(blockYPosition - BMUYPosition);
	float learningDistance = 0;
	
	if(manhattanDist <= (int) neighborhoodRadius){
		learningDistance = exp( (-1 * manhattanDist * manhattanDist) / (2 * neighborhoodRadius * neighborhoodRadius));
		
		for (int i = 0; i < VLAD_CENTERS; i++){
			d_somGrid[blockIdx.x*ORB_DESCRIPTOR_DIMENSION*VLAD_CENTERS + i*ORB_DESCRIPTOR_DIMENSION + threadIdx.x] = 
				d_somGrid[blockIdx.x*ORB_DESCRIPTOR_DIMENSION*VLAD_CENTERS + i*ORB_DESCRIPTOR_DIMENSION + threadIdx.x] + learningDistance * currentLearningRate *
				(input[i * VLAD_CENTERS + threadIdx.x ] - d_somGrid[blockIdx.x*ORB_DESCRIPTOR_DIMENSION*VLAD_CENTERS + i*ORB_DESCRIPTOR_DIMENSION + threadIdx.x]);

		}
		
	}
	
}

__host__ void updateSOMGPU(SOM &som, float *h_input, int indexOfBMU){
	float *d_input;

	cudaError_t error;

	error = cudaMalloc((void **)&d_input, VLAD_CENTERS * ORB_DESCRIPTOR_DIMENSION * sizeof(float));

	if (error != ::cudaSuccess)
	{
		std::cerr << "Could not allocate sample input descriptor: " << error << std::endl;
	}

	error = cudaMemcpy(d_input, h_input, VLAD_CENTERS * ORB_DESCRIPTOR_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);

	if (error != ::cudaSuccess)
	{
		cerr << "Could not copy inpýt descriptor to device: " << error << endl;
	}

	som.neighborhoodRadius = som.mapRadius * exp( -(float) som.currentIterarion / som.timeConst );
	som.currentLearningRate = som.initialLearningRate * exp(-(float)som.currentIterarion/som.maxIterationNum); // current learing rate is decreasing with iterations.

	/*
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);	
	*/
	updateSOMKernel <<<SOM_GRID_SIZE*SOM_GRID_SIZE, 32 >>>(som.d_somGrid, d_input, indexOfBMU, som.neighborhoodRadius, som.currentLearningRate);
	/*
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("updateSOMGPU time: %f\n", milliseconds);
	*/

	som.currentIterarion++;
}