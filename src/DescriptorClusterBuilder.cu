#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda.h>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <stdint.h>
#include <windows.h>

#define DEBUG 0

using namespace std;

__global__ void resetClusters(short* clusters, int nClusters, short value){
	int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (globalThreadId < nClusters){
		clusters[globalThreadId] = value;
	}
}

__global__ void clusterKeypoints(cv::cuda::PtrStepSz<short4> boundingRects, float *keypointsX, float *keypointsY, int nKeypoints, short* clusters){
	extern __shared__ short4 _boundingRects[];

	int keypointsPerBlock = blockDim.x / boundingRects.rows; // 128
	int keypointsStartIdx = keypointsPerBlock * blockIdx.x; // 384
	int keypointIdx = (keypointsStartIdx + threadIdx.x / boundingRects.rows); // 501
	int rectIdx = threadIdx.x % boundingRects.rows; // 1

	/*if (rectIdx == 0){
		printf("%d,", clusters[keypointIdx]);
	}

	
	if (threadIdx.x < boundingRects.rows)
	{
		printf("Keypoints: %d", nKeypoints);
		printf("rect %d: (%d,%d | %d,%d)\n", threadIdx.x, boundingRects(threadIdx.x, 0).x, boundingRects(threadIdx.x, 0).y, boundingRects(threadIdx.x, 0).z, boundingRects(threadIdx.x, 0).w);
	}*/

	// cooperativeley load bounding rects into shared memory
	// each thread reads 8-bytes from global into shared memory - should result in one transaction per half-warp
	if (threadIdx.x < boundingRects.rows)
	{
		_boundingRects[threadIdx.x] = boundingRects(threadIdx.x, 0);
	}
	
	__syncthreads();

	if (keypointIdx < min(keypointsStartIdx + keypointsPerBlock, nKeypoints))
	{
		float keypointX = keypointsX[keypointIdx];
		float keypointY = keypointsY[keypointIdx];

		//printf("Thread %d keypointIdx %d rectIdx %d\n", threadIdx.x, keypointIdx, rectIdx);
		// boundingRect : (x=topLeft.x, y=topLeft.y, z=bottomRight.x, w=bottomRight.y)
		if (keypointX >= _boundingRects[rectIdx].x && keypointX <= _boundingRects[rectIdx].z &&
			keypointY >= _boundingRects[rectIdx].y && keypointY <= _boundingRects[rectIdx].w)
		{
			assert(clusters[keypointIdx] == -1);
			clusters[keypointIdx] = rectIdx;
			//printf("keypointIdx %d = %d\n", keypointIdx, rectIdx);
		}
		/*else
		{
			printf("keypoint %d (%.2f, %.2f) not in (%d,%d | %d,%d)\n", keypointIdx, keypointX, keypointY, _boundingRects[rectIdx].x, _boundingRects[rectIdx].y, _boundingRects[rectIdx].z, _boundingRects[rectIdx].w);
		}*/
	}
}

inline int divUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void clusterKeypoints_gpu(cv::cuda::PtrStepSz<short4> boundingRects, float *keypointsX, float *keypointsY, int numPoints, short* clusters, cudaStream_t stream)
{
	dim3 block(256);

	assert(boundingRects.rows < 256);

	dim3 grid;
	grid.x = divUp(numPoints * boundingRects.rows, block.x);

	const int bytesSharedMem = sizeof(short4) * boundingRects.rows;

#if DEBUG==1
	cout << "Launching kernel with grid dimensions " << grid.x << ", block dimensions " << block.x << " and " << bytesSharedMem  << " bytes of shared mem per block" << endl;
	cout << "Num keypoints: " << numPoints << endl;
	cout << "Num bounding rects: " << boundingRects.rows << endl;
#endif

	clusterKeypoints << <grid, block, bytesSharedMem, stream >> >(boundingRects, keypointsX, keypointsY, numPoints, clusters);

	cudaSafeCall(cudaGetLastError());

	if (stream == 0)
	{
		cudaSafeCall(cudaDeviceSynchronize());
	}
}

void resetClusters_gpu(short* clusters, int nClusters, short value, cudaStream_t stream)
{
	dim3 block(256);
	dim3 grid;
	grid.x = divUp(nClusters, block.x);

	resetClusters <<<grid, block, 0, stream>>>(clusters, nClusters, value);
}

template<typename T> void printMatrix(const cv::Mat& a)
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

void printByteMatrix(const cv::Mat& a)
{
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			cout << (int) a.at<uint8_t>(i, j) << ",";
		}
		cout << endl;
	}
}

void clusterORBDescriptors(const cv::InputArray boundingRects, const cv::InputArray keypoints, const cv::InputArray desc, cv::OutputArray clusters, std::vector<std::pair<int, cv::Mat>> perObjectDescriptors, cv::cuda::Stream& stream)
{
	assert(boundingRects.rows() == perObjectDescriptors.size());
	cudaStream_t cudaStream = cv::cuda::StreamAccessor::getStream(stream);
	cv::cuda::ensureSizeIsEnough(cv::Size(1, keypoints.cols()), CV_16U, clusters);
	
	resetClusters_gpu(clusters.getGpuMat().ptr<short>(), keypoints.cols(), -1, cudaStream);
	clusterKeypoints_gpu(boundingRects.getGpuMat(), keypoints.getGpuMat().ptr<float>(cv::cuda::ORB::X_ROW), keypoints.getGpuMat().ptr<float>(cv::cuda::ORB::Y_ROW), keypoints.cols(), clusters.getGpuMat().ptr<short>(), cudaStream);

	cv::Mat hostClusters;
	clusters.getGpuMat().download(hostClusters, stream);
	int *descriptorCounts = new int[perObjectDescriptors.size()]();

	for (int i = 0; i < perObjectDescriptors.size(); i++)
	{
		std::pair<int, cv::Mat> elem = perObjectDescriptors.at(i);
		elem.second.resize(desc.rows());
	}

	stream.waitForCompletion();

	for (int descIdx = 0; descIdx < desc.rows(); descIdx++)
	{
		short clusterIdx = hostClusters.at<short>(descIdx);
		if (clusterIdx >= 0)
		{
			cv::Mat target = perObjectDescriptors.at(clusterIdx).second;
			desc.getMat().row(descIdx).copyTo(target.row(descriptorCounts[clusterIdx]++));
		}
	}
	
	for (int objectIdx = 0; objectIdx < perObjectDescriptors.size(); objectIdx++)
	{
		perObjectDescriptors.at(objectIdx).second.resize(descriptorCounts[objectIdx]);
	}

	delete[] descriptorCounts;
}