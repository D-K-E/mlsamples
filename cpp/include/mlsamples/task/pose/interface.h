#ifndef INTERFACE_H
#define INTERFACE_H

#include <torch/torch.h>
//

#include <filesystem>
#include <tuple>
#include <vector>
namespace mlsamples {
struct Keypoints {
  Keypoints() = delete;
  Keypoints(torch::Tensor f,
            std::vector<std::pair<int, int>> ps);
  torch::Tensor frame;
  std::vector<std::pair<int, int>> keypoints;
};

class PoseEstimator {
public:
  virtual std::vector<Keypoints>
  run(std::filesystem::path video) = 0;
};
} // namespace mlsamples
#endif
