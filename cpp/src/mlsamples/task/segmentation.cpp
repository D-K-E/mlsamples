// implement detection related stuff
#include "yoloutils.h"
#include <mlsamples/task/segmentation/seginterface.h>
#include <mlsamples/task/segmentation/segyolo.h>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <torch/torch.h>
#include <vector>

namespace mlsamples {
namespace segmentation {
//
// segmentation
Mask::Mask(
    cv::Mat f,
    const std::vector<std::vector<std::pair<int, int>>> &m)
    : frame(f), masks(m) {}

// Impl definition
struct Yolo::Impl {
  Impl() {
    auto temp =
        std::make_unique<YoloV8Model>(Task::SEGMENTATION);
    model = std::move(temp);
  }
  ~Impl() = default;
  std::unique_ptr<YoloV8Model> model;
};

Yolo::Yolo() {
  auto impl_p = std::make_unique<Impl>();
  impl = std::move(impl_p);
}
Yolo::~Yolo() = default;

std::vector<Mask> Yolo::run(std::filesystem::path video) {
  //
  std::vector<Mask> masks;

  //
  std::string vid_s = video.string();
  cv::VideoCapture vcapt(vid_s);
  while (vcapt.isOpened()) {
    cv::Mat frame;
    bool isSuccess = vcapt.read(frame);
    // If frames are present, show it
    if (isSuccess == true) {
      // infer with model
      std::vector<float> input_t =
          yolo_preprocess_image(frame);
      torch::Tensor input = torch::from_blob(
          input_t.data(),
          {1, kYOLO_CHANNEL_, kYOLO_NETWORK_HEIGHT_,
           kYOLO_NETWORK_WIDTH_},
          torch::kFloat32);
      input = input.to(torch::kCPU);
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(input);
      auto [outputs, shapes] = impl->model->infer(inputs);
      cv::Size frame_size(frame.cols, frame.rows);
      std::vector<std::vector<std::pair<int, int>>>
          results =
              yolo_postprocess(outputs, shapes, frame_size);
      Mask d(frame, results);
      masks.push_back(d);
    }

    // If frames are not there, close it
    if (isSuccess == false) {
      break;
    }
  }
  return masks;
}
} // namespace segmentation

}; // namespace mlsamples
