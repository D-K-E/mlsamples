#ifndef DRAWER_H
#define DRAWER_H
// basic drawing functions
#include <mlsamples/task/detection/interface.h>
#include <mlsamples/task/pose/interface.h>
#include <mlsamples/task/segmentation/interface.h>
#include <opencv2/core.hpp>
#include <vector>
namespace mlsamples {

std::vector<cv::Mat> draw(const std::vector<Detection> &ds);
// std::vector<cv::Mat> draw(const std::vector<Keypoints>
// &ks); std::vector<cv::Mat> draw(const std::vector<Masks>
// &ks);
} // namespace mlsamples

#endif
