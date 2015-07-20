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

// kernel for initializing clusters array - all array elements are set to value
__global__ void resetClusters(short* clusters, int nClusters, short value){
	int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (globalThreadId < nClusters){
		clusters[globalThreadId] = value;
	}
}

// kernel for assigning keypoints to bounding rectangles
__global__ void clusterKeypoints(cv::cuda::PtrStepSz<short4> boundingRects, float *keypointsX, float *keypointsY, int nKeypoints, short* clusters){
	extern __shared__ short4 _boundingRects[];

	int keypointsPerBlock = blockDim.x / boundingRects.rows;
	int keypointsStartIdx = keypointsPerBlock * blockIdx.x;
	int keypointIdx = (keypointsStartIdx + threadIdx.x / boundingRects.rows);
	int rectIdx = threadIdx.x % boundingRects.rows;

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

		// each thread matches one keypoint to one bounding rectangle
		if (keypointX >= _boundingRects[rectIdx].x && keypointX <= _boundingRects[rectIdx].z &&
			keypointY >= _boundingRects[rectIdx].y && keypointY <= _boundingRects[rectIdx].w)
		{
#if DEBUG==1
			// only enable for debug because assert affects performance
			assert(clusters[keypointIdx] == -1);
#endif
			// update cluster index if containment is detected
			clusters[keypointIdx] = rectIdx;
		}
	}
}

inline int divUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void clusterKeypoints_gpu(cv::cuda::PtrStepSz<short4> boundingRects, float *keypointsX, float *keypointsY, int numPoints, short* clusters, cudaStream_t stream)
{
	dim3 block(256);

	// we make an assumption on the maximum number of bounding rectangles per frame
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
	
	// initialize cluster array
	resetClusters_gpu(clusters.getGpuMat().ptr<short>(), keypoints.cols(), -1, cudaStream);
	// cluster keypoints
	clusterKeypoints_gpu(boundingRects.getGpuMat(), keypoints.getGpuMat().ptr<float>(cv::cuda::ORB::X_ROW), keypoints.getGpuMat().ptr<float>(cv::cuda::ORB::Y_ROW), keypoints.cols(), clusters.getGpuMat().ptr<short>(), cudaStream);

	// download clusters
	cv::Mat hostClusters;
	clusters.getGpuMat().download(hostClusters, stream);
	int *descriptorCounts = new int[perObjectDescriptors.size()]();

	// setup Mats to hold the clustered descriptors
	for (int i = 0; i < perObjectDescriptors.size(); i++)
	{
		std::pair<int, cv::Mat> elem = perObjectDescriptors.at(i);
		elem.second.resize(desc.rows());
	}

	stream.waitForCompletion();

	// copy each descriptor to one of the descriptor Mats according to the cluster index
	for (int descIdx = 0; descIdx < desc.rows(); descIdx++)
	{
		short clusterIdx = hostClusters.at<short>(descIdx);
		if (clusterIdx >= 0)
		{
			cv::Mat target = perObjectDescriptors.at(clusterIdx).second;
			desc.getMat().row(descIdx).copyTo(target.row(descriptorCounts[clusterIdx]++));
		}
	}
	
	// resize the pessimistically allocated descriptor matrices
	for (int objectIdx = 0; objectIdx < perObjectDescriptors.size(); objectIdx++)
	{
		perObjectDescriptors.at(objectIdx).second.resize(descriptorCounts[objectIdx]);
	}

	delete[] descriptorCounts;
}