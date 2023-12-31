#ifndef SEGYOLO_H
#define SEGYOLO_H

#include "seginterface.h"

//
#include <filesystem>
#include <memory>

namespace mlsamples {
namespace segmentation {

class Yolo : public Segmenter {
public:
  Yolo();
  ~Yolo();
  virtual std::vector<Mask>
  run(std::filesystem::path video) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace segmentation
} // namespace mlsamples
#endif
