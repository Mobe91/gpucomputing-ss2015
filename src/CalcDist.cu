#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda.h>
#include "Constants.h"
#include <stdint.h>
#include <math_constants.h>
#include <device_functions.h>
#include "CalcDist.cuh"



__global__ void calcDist(float *map, float *input, float *result){
	int tid = threadIdx.x;


	__shared__ float neuron_vector[ORB_DESCRIPTOR_DIMENSION];//a single vector of the neuron

	__shared__ float dist[ORB_DESCRIPTOR_DIMENSION*VLAD_CENTERS]; //distance between values of 1 vector to all neuron vectors
	__shared__ float2 sum[VLAD_CENTERS*VLAD_CENTERS]; //summed up distances

	float distance = 1;

	for (int i = 0; i < VLAD_CENTERS; i++){
		neuron_vector[tid] = map[blockIdx.x*ORB_DESCRIPTOR_DIMENSION*VLAD_CENTERS + i*VLAD_CENTERS + tid]; //load one vector of the neuron
		//__syncthreads();

		//calculate distance matrix ( one neuron vector to all input vectors
		for (int j = 0; j < VLAD_CENTERS; j++){
			dist[j*ORB_DESCRIPTOR_DIMENSION + tid] = abs(neuron_vector[tid] - input[j*VLAD_CENTERS + tid]);
		}
		//__syncthreads();

		//sum the individual distance values
		//unrolled reduction
		//TODO half of the threads are idle at the beginning
		for (int j = 0; j < VLAD_CENTERS; j++){
			if (tid < 16){
				dist[tid] += dist[tid + 16];
				dist[tid] += dist[tid + 8];
				dist[tid] += dist[tid + 4];
				dist[tid] += dist[tid + 2];
				dist[tid] += dist[tid + 1];
			}
			//__syncthreads();
			int target = i*VLAD_CENTERS + j;
			if (tid == 0){
				sum[target].x = dist[0];
				sum[target].y = j;
			}
		}

	}
	//Here the really inefficient part starts
	//complexity ~ O(VLAD_CENTER²)
	for (int i = 0; i < VLAD_CENTERS; i++){
		//find the best matching vector
		//reduction would be slower because we only have ~5 centers
		//TODO only 1 thread active
		float2 best;
		best.x = CUDART_INF_F;
		if (tid == 0){
			for (int j = 0; j < VLAD_CENTERS; j++) {
				//printf("val %f\n", sum[i*VLAD_CENTERS + tid].x);
				if (sum[i*VLAD_CENTERS + tid].x < best.x){
					//printf("new best %f\n", sum[i*VLAD_CENTERS + tid].x);
					best.x = sum[i*VLAD_CENTERS + j].x;
					best.y = j;
				}
			}
			distance += best.x;
		}
		//__syncthreads();

		//best match found --> do not check this vector
		//set all sums for this vector to max float
		for (int j = 0; j < (VLAD_CENTERS*VLAD_CENTERS) / blockDim.x; j++){
			int k = j* blockDim.x + tid;
			if (k < VLAD_CENTERS*VLAD_CENTERS){
				if (sum[k].y == best.y) sum[k].x = CUDART_INF_F;
			}
		}
	}
	if (tid == 0){
		*result = distance;
	}
}

__host__ float calcDistGPU(float *som, float *input){

	float *d_input;
	float *d_result;
	cudaError_t error;

	cudaMalloc((void **)&d_input, VLAD_CENTERS * ORB_DESCRIPTOR_DIMENSION * sizeof(float));
	cudaMalloc((void **)&d_result, sizeof(float));
	cudaMemcpy(d_input, input, VLAD_CENTERS * ORB_DESCRIPTOR_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);	//copy input vlad matrix to the device

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	calcDist << <SOM_GRID_SIZE*SOM_GRID_SIZE, 32 >> >(som, d_input, d_result);
	cudaEventRecord(stop);

	float result;
	cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time: %f\n", milliseconds);

	return result;
}