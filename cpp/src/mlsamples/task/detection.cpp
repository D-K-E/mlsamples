// implement detection related stuff
#include <mlsamples/loader.h>
#include <mlsamples/task/detection/interface.h>
#include <mlsamples/task/detection/yolo.h>
#include <opencv2/videoio.hpp>
#include <torch/torch.h>
#include <vector>

namespace mlsamples {
namespace detection {
//
// detection
Detection::Detection(torch::Tensor f, torch::Tensor b)
    : frame(f), bboxes(b) {}

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
    model.eval();
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
  std::vector<Detection> detections;

  //
  std::string vid_s = video.string();
  cv::VideoCapture vcapt(vid_s);
  while (vcapt.isOpened()) {
    cv::Mat frame;
    bool isSuccess = vid_capture.read(frame);
    // If frames are present, show it
    if (isSuccess == true) {
      // infer with model
      // convert frame to float, then to tensor
      c10::IValue v{vid_s};
    }

    // If frames are not there, close it
    if (isSuccess == false) {
      break;
    }
  }
  // read video
  std::vector<c10::IValue> vs{v};
  auto results = impl->model(vs);
  std::cout << video << std::endl;
  //  for (const auto &result : results) {
  //  }
  return detections;
}
} // namespace detection

}; // namespace mlsamples
