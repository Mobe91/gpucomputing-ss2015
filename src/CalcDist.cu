#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda.h>
#include "Constants.h"
#include <stdint.h>
#include <math_constants.h>
#include <device_functions.h>
#include "CalcDist.cuh"

///////////////////////////////////////////
// NOT USED
///////////////////////////////////////////

__global__ void calcDist(float *map, float *input, float *result){
	int tid = threadIdx.x;
	int nid = tid >> 5; //number of neuron inside the block
	int rowid = tid & (32 - 1);//id inside the neuron, tid modulo 32

	__shared__ float neuron_vector[ORB_DESCRIPTOR_DIMENSION];//a single vector of the neuron

	__shared__ float dist[ORB_DESCRIPTOR_DIMENSION*VLAD_CENTERS]; //distance between values of 1 vector to all neuron vectors
	__shared__ float2 sum[VLAD_CENTERS*VLAD_CENTERS]; //summed up distances

	__shared__ float distance;
	if (tid == 0) distance = 0;

	int addr; //universal address variable;

	for (int i = 0; i < VLAD_CENTERS; i++){
		neuron_vector[tid] = map[blockIdx.x*ORB_DESCRIPTOR_DIMENSION*VLAD_CENTERS + i*ORB_DESCRIPTOR_DIMENSION + tid]; //load one vector of the neuron
		//printf("block: %d, neuron tid %d: %f\n", blockIdx.x, tid, neuron_vector[i*VLAD_CENTERS + tid]);
		//printf("block: %d, input tid %d: %f\n", blockIdx.x, tid, input[i*VLAD_CENTERS + tid]);
		//__syncthreads();

		//calculate distance matrix ( one neuron vector to all input vectors
		for (int j = 0; j < VLAD_CENTERS; j++){
			addr = j*ORB_DESCRIPTOR_DIMENSION + tid;
			dist[addr] = abs(neuron_vector[tid] - input[addr]);
			//printf("%d: %g from %g - %g\n", addr, dist[addr], neuron_vector[tid], input[addr]);
		}
		//__syncthreads();

		//sum the individual distance values
		//unrolled reduction
		//TODO half of the threads are idle at the beginning
		for (int j = 0; j < VLAD_CENTERS; j++){
			int addr = j * ORB_DESCRIPTOR_DIMENSION + tid;
			if (tid < 16){
				dist[addr] += dist[addr + 16];
				dist[addr] += dist[addr + 8];
				dist[addr] += dist[addr + 4];
				dist[addr] += dist[addr + 2];
				dist[addr] += dist[addr + 1];
			}
			//__syncthreads();
			
			if (tid == 0){
				sum[i*VLAD_CENTERS + j].x = dist[addr];
				sum[i*VLAD_CENTERS + j].y = j;
				//printf("new sum value at %d is %f from %d\n", addr, sum[i*VLAD_CENTERS + j].x, (int)sum[i*VLAD_CENTERS + j].y);
			}
		}

	}
	//Here the really inefficient part starts
	//complexity ~ O(VLAD_CENTER²)
	for (int i = 0; i < VLAD_CENTERS; i++){
		//find the best matching vector
		//reduction would be slower because we only have ~5 centers
		//TODO only 1 thread active
		__shared__ float bestval;
		__shared__ int best;
		if (tid == 0){

			bestval = CUDART_INF_F;
			for (int j = 0; j < VLAD_CENTERS; j++) {
				//printf("val %f\n", sum[i*VLAD_CENTERS + j].x);
				if (sum[i*VLAD_CENTERS + j].x < bestval){
					//printf("new best is %d with %f\n",j, sum[i*VLAD_CENTERS + j].x);
					bestval = sum[i*VLAD_CENTERS + j].x;
					best = j;
				}
			}
			//printf("match %d %d: dist = %f\n", i, best, bestval);
			distance += bestval;
		}

		//best match found --> do not check this vector
		//set all sums for this vector to max float
		for (int j = 0; j < (VLAD_CENTERS*VLAD_CENTERS) / blockDim.x + 1; j++){

			int k = j* blockDim.x + tid;
			if (k < VLAD_CENTERS*VLAD_CENTERS){
				if ((int)sum[k].y == best){
					sum[k].x = CUDART_INF_F;
				}
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

__host__ float calcDistGPU2(float *inputA, float *inputB){

	float *d_inputA, *d_inputB;
	float *d_result;
	cudaError_t error;

	error = cudaMalloc((void **)&d_inputA, VLAD_CENTERS * ORB_DESCRIPTOR_DIMENSION * sizeof(float));
	error = cudaMalloc((void **)&d_inputB, VLAD_CENTERS * ORB_DESCRIPTOR_DIMENSION * sizeof(float));
	error = cudaMalloc((void **)&d_result, sizeof(float));
	error = cudaMemcpy(d_inputA, inputA, VLAD_CENTERS * ORB_DESCRIPTOR_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);	//copy input vlad matrix to the device
	error = cudaMemcpy(d_inputB, inputB, VLAD_CENTERS * ORB_DESCRIPTOR_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	calcDist << <1, 32 >> >(d_inputA, d_inputB, d_result); //SOM_GRID_SIZE*SOM_GRID_SIZE
	cudaEventRecord(stop);

	float result;
	error = cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("GPU time: %f\n", milliseconds);

	return result;
}