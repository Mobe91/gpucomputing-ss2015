#include <opencv2/core.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <cuda_runtime.h>

void clusterORBDescriptors(const cv::InputArray boundingRects, const cv::InputArray keypoints, const cv::InputArray desc, cv::OutputArray clusters, std::vector<std::pair<int, cv::Mat>> perObjectDescriptors, cv::cuda::Stream& stream);

