#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda.h>
#include <opencv2/core/cuda/common.hpp>

#include "Constants.h"
#include "CalcDist.cuh"

#include <stdint.h>
#include <math_constants.h>
#include <device_functions.h>

///////////////////////////////////////////
// NOT USED
///////////////////////////////////////////

#define DEBUG 1

#define THREADS_PER_BLOCK 128		//the maximum number of threads is limited by shared memory
#define IMAGES_PER_BLOCK (THREADS_PER_BLOCK/32)

__global__ void calcDist(float *input, float *result, int image_cnt){
	int tid = threadIdx.x;
	int iid = threadIdx.y; //number of image inside the block
	int sid = blockDim.y * blockIdx.x + iid; //number of the source image in this row


	__shared__ float image_vector[ORB_DESCRIPTOR_DIMENSION*IMAGES_PER_BLOCK];//a single descriptor vector of the image

	__shared__ float dist[ORB_DESCRIPTOR_DIMENSION*DESC_CENTERS*IMAGES_PER_BLOCK]; //distance between values of 1 vector to all neuron vectors
	__shared__ float2 sum[DESC_CENTERS*DESC_CENTERS*IMAGES_PER_BLOCK]; //summed up distances

	__shared__ float distance[IMAGES_PER_BLOCK];
	if (tid == 0) distance[iid] = 0;

	int addr; //universal address variable;

	 
	//int sum_offset = iid* DESC_CENTERS * DESC_CENTERS; //this value is needed alot --> only compute it once

	for (int i = 0; i < VLAD_CENTERS; i++){
		int img_offset = iid * ORB_DESCRIPTOR_DIMENSION;
		image_vector[img_offset + tid] = input[sid*ORB_DESCRIPTOR_DIMENSION*DESC_CENTERS + i*ORB_DESCRIPTOR_DIMENSION + tid]; //load one vector of the neuron
		//printf("block: %d, neuron tid %d: %f\n", blockIdx.x, tid, image_vector[i*VLAD_CENTERS + tid]);
		//printf("block: %d, input tid %d: %f\n", blockIdx.x, tid, input[i*VLAD_CENTERS + tid]);
		__syncthreads();

		//calculate distance matrix (one neuron vector to all input vectors)
		for (int j = 0; j < DESC_CENTERS; j++){
			addr = j*ORB_DESCRIPTOR_DIMENSION + tid + ORB_DESCRIPTOR_DIMENSION*DESC_CENTERS*iid;
			dist[addr] = pow(image_vector[img_offset + tid] - input[blockIdx.y * ORB_DESCRIPTOR_DIMENSION*DESC_CENTERS + j*ORB_DESCRIPTOR_DIMENSION + tid], 2);
			//printf("%d: %g from %g - %g\n", addr, dist[addr], image_vector[tid], input[addr]);
		}
		__syncthreads();

		//sum the individual distance values
		//unrolled reduction
		//TODO half of the threads are idle at the beginning
		for (int j = 0; j < DESC_CENTERS; j++){
			int addr = ORB_DESCRIPTOR_DIMENSION*DESC_CENTERS*iid + j * ORB_DESCRIPTOR_DIMENSION + tid; //image offset + row offset + thread offset
			if (tid < 16){
				dist[addr] += dist[addr + 16];
				dist[addr] += dist[addr + 8];
				dist[addr] += dist[addr + 4];
				dist[addr] += dist[addr + 2];
				dist[addr] += dist[addr + 1];
			}

			if (tid == 0){
				int dest = iid* DESC_CENTERS * DESC_CENTERS + i*DESC_CENTERS + j;
				sum[dest].x = dist[addr];
				sum[dest].y = j;
				//printf("new sum value at %d is %f from %d\n", addr, sum[i*VLAD_CENTERS + j].x, (int)sum[i*VLAD_CENTERS + j].y);
			}
		}

	}
	//Here the really inefficient part starts
	//complexity ~ O(VLAD_CENTER²)
	for (int i = 0; i < DESC_CENTERS; i++){
		//find the best matching vector
		//reduction would not work, because we also need the position of the best match
		//TODO only IMAGES_PER_THREAD number of threads active
		__shared__ float bestval[IMAGES_PER_BLOCK];
		__shared__ int best[IMAGES_PER_BLOCK];
		if (tid == 0){

			bestval[iid] = CUDART_INF_F;
			for (int j = 0; j < DESC_CENTERS; j++) {
				//printf("val %f\n", sum[i*VLAD_CENTERS + j].x);
				int addr = iid* DESC_CENTERS * DESC_CENTERS + i*DESC_CENTERS + j;
				if (sum[addr].x < bestval[iid]){
					//printf("new best is %d with %f\n",j, sum[i*VLAD_CENTERS + j].x);
					bestval[iid] = sum[addr].x;
					best[iid] = j;
				}
			}
			//printf("match %d %d: dist = %f\n", i, best, bestval);
			distance[iid] += bestval[iid];
		}

		//best match found --> do not check this vector
		//set all sums for this vector to max float
		for (int j = 0; j < (DESC_CENTERS*DESC_CENTERS) / blockDim.x + 1; j++){

			int k = j* blockDim.x + tid;
			if (k < DESC_CENTERS*DESC_CENTERS){
				if ((int)sum[k + iid* DESC_CENTERS * DESC_CENTERS].y == best[iid]){
					sum[k + iid* DESC_CENTERS * DESC_CENTERS].x = CUDART_INF_F;
				}
			}
		}
	}
	if (tid == 0){
		result[gridDim.y * blockIdx.y + sid] = distance[iid];
	}
}

__host__ float* calcDistGPU(float *input, int image_cnt){

	float *d_input;
	float *d_result;
	

	cudaMalloc((void **)&d_input, DESC_CENTERS * ORB_DESCRIPTOR_DIMENSION * image_cnt * sizeof(float));
	cudaMalloc((void **)&d_result, image_cnt * image_cnt * sizeof(float));
	cudaMemcpy(d_input, input, DESC_CENTERS * ORB_DESCRIPTOR_DIMENSION * image_cnt * sizeof(float), cudaMemcpyHostToDevice);	//copy input images to the device

	cudaSafeCall(cudaGetLastError());

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 grid,block;
	grid.x = image_cnt / IMAGES_PER_BLOCK; // image cound divided by images per block
	grid.y = image_cnt;

	block.x = 32;
	block.y = IMAGES_PER_BLOCK;
	
	cudaEventRecord(start);
	calcDist << <grid, block >> >(d_input, d_result, image_cnt);
	cudaSafeCall(cudaGetLastError());
	cudaEventRecord(stop);

	std::vector<float> result;
	result.resize(image_cnt * image_cnt);
	cudaMemcpy(result.data(), d_result, sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	if(DEBUG == 1) printf("GPU time: %f\n", milliseconds);

	return result.data();
}

