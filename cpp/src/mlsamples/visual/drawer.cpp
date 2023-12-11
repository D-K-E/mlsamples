#include "drawer.h"

// other stuff
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>
using namespace torch::indexing;

namespace mlsamples {
//
std::vector<cv::Mat>
draw(const std::vector<detection::Detection> &ds) {
  //
  std::vector<cv::Mat> results;
  for (auto d : ds) {
    cv::Mat frame = d.frame;
    std::vector<cv::Rect> boxes = d.bboxes;
    for (int i = 0; i < boxes.size(); ++i) {
      auto box = boxes[i];
      cv::rectangle(frame, box, cv::Scalar(255,0,255));
    }
    results.push_back(frame);
  }
  return results;
}

std::vector<cv::Mat>
draw(const std::vector<segmentation::Mask> &masks) {

  std::vector<cv::Mat> results;
  auto to_p = [](const std::pair<int, int> &point) {
    int x = point.first;
    int y = point.second;
    auto p = cv::Point(x, y);
    return p;
  };
  for (auto mask : masks) {
    cv::Mat frame = mask.frame;
    std::vector<std::vector<std::pair<int, int>>>
        ms_per_frame = mask.masks;
    cv::Mat temp(frame.rows, frame.cols, CV_8UC3,
                 cv::Scalar(0));
    for (int i = 0; i < ms_per_frame.size(); ++i) {
      std::vector<std::pair<int, int>> points =
          ms_per_frame[i];
      std::vector<cv::Point> ps(points.size());
      std::transform(points.begin(), points.end(),
                     ps.begin(), to_p);
      cv::fillConvexPoly(temp, ps.data(), ps.size(),
                         cv::Scalar(255, 0, 255));
    }
    cv::Mat result;
    cv::addWeighted(frame, 0.6, temp, 0.4, 0, result);
    results.push_back(result);
  }
  return results;
}
} // namespace mlsamples
