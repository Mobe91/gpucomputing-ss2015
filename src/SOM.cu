#include "SOM.h"
#include "Constants.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda.h>
#include <time.h>

using namespace std;

SOM::SOM(const int gridSize) : gridSize(gridSize) { }

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
				h_somGrid + y * sampleVectorRows * sampleVectorCols * gridSize + x * sampleVectorCols * VLAD_CENTERS,
				sampleVectors + sampleVectorIdx * sampleVectorCols * VLAD_CENTERS,
				sampleVectorCols * VLAD_CENTERS * sizeof(float)
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