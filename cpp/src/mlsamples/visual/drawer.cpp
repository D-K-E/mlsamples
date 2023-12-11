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
      cv::rectangle(frame, box, cv::Scalar(0));
    }
    results.push_back(frame);
  }
  return results;
}

std::vector<cv::Mat> draw(const std::vector<Masks> &masks) {

  std::vector<cv::Mat> results;
  auto to_p = [](const std::pair<int, int> &point) {
    return cv::Point(point.first, point.second);
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
      std::vector<cv::Point> ps;
      std::transform(points.begin(), points.end(),
                     ps.begin(), to_p);
      cv::fillConvexPoly(temp, ps.data(), cv::Scalar(0));
    }
    cv::Mat result;
    cv::addWeighted(frame, 0.8, temp, 0.2, 0, result);
    results.push_back(result);
  }
}
} // namespace mlsamples
