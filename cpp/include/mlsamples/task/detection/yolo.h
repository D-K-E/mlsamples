#ifndef YOLO_H
#define YOLO_H
#include "interface.h"

//
#include <filesystem>
#include <memory>
namespace mlsamples {

namespace detection {

class Yolo : public Detector {
public:
  Yolo();
  ~Yolo();
  virtual std::vector<Detection>
  run(std::filesystem::path video) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace detection
} // namespace mlsamples
#endif
