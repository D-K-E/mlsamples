#ifndef SEGINTERFACE_H
#define SEGINTERFACE_H

#include <opencv2/core.hpp>
//

#include <filesystem>
#include <tuple>
#include <vector>
namespace mlsamples {
namespace segmentation {

struct Mask {
  Mask() = delete;
  Mask(cv::Mat f,
       const std::vector<std::vector<std::pair<int, int>>>
           &m);

  cv::Mat frame;
  std::vector<std::vector<std::pair<int, int>>> masks;
};

class Segmenter {
public:
  virtual std::vector<Mask>
  run(std::filesystem::path video) = 0;
};
} // namespace segmentation
} // namespace mlsamples
#endif
