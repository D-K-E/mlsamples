#ifndef INTERFACE_H
#define INTERFACE_H

#include <torch/torch.h>
//

#include <filesystem>
#include <tuple>
#include <vector>
namespace mlsamples {
struct Mask {
  Mask() = delete;
  Mask(torch::Tensor f,
       std::vector<std::vector<std::pair<int, int>>> m);
  torch::Tensor frame;
  std::vector<std::vector<std::pair<int, int>>> masks;
};

class Segmenter {
public:
  virtual std::vector<Mask>
  run(std::filesystem::path video) = 0;
};
} // namespace mlsamples
#endif
