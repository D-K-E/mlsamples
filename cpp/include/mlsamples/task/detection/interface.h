#ifndef INTERFACE_H
#define INTERFACE_H

#include <torch/torch.h>
//

#include <filesystem>
#include <vector>
namespace mlsamples {
namespace detection {
struct Detection {
  Detection() = delete;
  Detection(torch::Tensor f, torch::Tensor b);
  torch::Tensor frame;
  torch::Tensor bboxes;
};

class Detector {
public:
  virtual std::vector<Detection>
  run(std::filesystem::path video) = 0;
};
} // namespace detection
} // namespace mlsamples
#endif
