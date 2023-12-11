// implement detection related stuff
#include "yoloutils.h"
#include <mlsamples/loader.h>
#include <mlsamples/task/detection/interface.h>
#include <mlsamples/task/detection/yolo.h>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <torch/torch.h>
#include <vector>

namespace mlsamples {
namespace detection {
//
// detection
Detection::Detection(cv::Mat f,
                     const std::vector<cv::Rect> &b)
    : frame(f), bboxes(b) {}

// Impl definition
struct Yolo::Impl {
  Impl() {
    auto temp =
        std::make_unique<YoloV8Model>(Task::DETECTION);
    model = std::move(temp);
  }
  ~Impl() = default;

  std::unique_ptr<YoloV8Model> model{nullptr};
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
      std::vector<cv::Rect> results =
          detection::yolo_postprocess(outputs, shapes,
                                      frame_size);
      Detection d(frame, results);
      detections.push_back(d);
    }

    // If frames are not there, close it
    if (isSuccess == false) {
      break;
    }
  }
  return detections;
}
} // namespace detection

}; // namespace mlsamples
