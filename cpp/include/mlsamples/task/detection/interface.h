#ifndef INTERFACE_H
#define INTERFACE_H

#include <torch/torch.h>
#include <opencv2/core.hpp>
//

#include <filesystem>
#include <vector>
namespace mlsamples {
namespace detection {
struct Detection {
  Detection() = delete;
  Detection(cv::Mat f, const std::vector<cv::Rect>& b);
  cv::Mat frame;
  std::vector<cv::Rect> bboxes;
};

class Detector {
public:
  virtual std::vector<Detection>
  run(std::filesystem::path video) = 0;
};
} // namespace detection
} // namespace mlsamples
#endif
