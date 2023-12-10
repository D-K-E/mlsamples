// implement detection related stuff
#include <mlsamples/loader.h>
#include <mlsamples/task/detection/interface.h>
#include <mlsamples/task/detection/yolo.h>
#include <torch/torch.h>
#include <vector>

namespace mlsamples {
//
// detection
Detection::Detection(torch::Tensor f, torch::Tensor b)
    : frame(f), bboxes(b) {}

Detection::~Detection() = default;

// Impl definition
struct Yolo::Impl {
  Impl() {
    std::filesystem::path p(ASSET_DIR);
    p = p.make_preferred();
    p /= std::filesystem::path("yolov8x.torchscript");
    if (!std::filesystem::exists(p)) {
      std::string m(
          "model yolov8x.torchscript can not be found: ");
      m += p.string();
      throw std::runtime_error(m.c_str());
    }
    model = mlsamples::load(p);
  }
  ~Impl() = default;

  torch::jit::Module model;
};

Yolo::Yolo() {
  auto impl_p = std::make_unique<Impl>();
  impl = std::move(impl_p);
}
Yolo::~Yolo() = default;

std::vector<Detection>
Yolo::run(std::filesystem::path video) {
  //
  std::string vid_s = video.string();
  auto results = impl->model(vid_s);
  std::vector<Detection> detections;
  for (const auto &result : results) {
  }
  return detections;
}

}; // namespace mlsamples
