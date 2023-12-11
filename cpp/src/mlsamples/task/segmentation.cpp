// implement detection related stuff
#include "yoloutils.h"
#include <mlsamples/task/segmentation/seginterface.h>
#include <mlsamples/task/segmentation/segyolo.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
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

std::vector<std::vector<std::pair<int, int>>>
to_mask_points(const SegOutput &results, cv::Mat frame) {
  cv::Mat masks;
  std::vector<cv::Mat> mask_channels;
  if (!results.bboxes.empty()) {
    cv::Mat roi_m = results.mask_proposals * results.segm;
    cv::Mat roi_t = roi_m.t();
    masks =
        roi_t.reshape(results.bboxes.size(), {160, 160});
    cv::split(masks, mask_channels);
  }
  //
  std::vector<std::vector<std::pair<int, int>>>
      frame_points;
  for (int i = 0; i < results.bboxes.size(); ++i) {
    std::vector<std::pair<int, int>> mask_points;
    cv::Mat temp;
    cv::exp(-mask_channels[i], temp);

    // Sigmoid
    cv::Mat mask = 1.0 / (1.0 + temp);

    mask = mask(results.roi);
    cv::resize(mask, mask, cv::Size(frame.cols, frame.rows),
               cv::INTER_NEAREST);
    cv::Mat m3 = mask(results.bboxes[i]) > 0.5f;

    const float mask_thresh = 0.5f;
    int x2 = results.bboxes[i].x + results.bboxes[i].width;
    int y2 = results.bboxes[i].y + results.bboxes[i].height;

    for (int w = 0; w < results.bboxes[i].width; ++w) {
      for (int h = 0; h < results.bboxes[i].height; ++h) {
        int x2 = results.bboxes[i].x + w;
        int y2 = results.bboxes[i].y + h;
        bool pix = m3.at<bool>(cv::Point(w, h));
        if (!pix) {
          std::pair<int, int> point =
              std::make_pair(x2, y2);
          mask_points.push_back(point);
        }
      }
    }
    frame_points.push_back(mask_points);
  }
  return frame_points;
}

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
      SegOutput results =
          yolo_postprocess(outputs, shapes, frame_size);
      std::vector<std::vector<std::pair<int, int>>> points =
          to_mask_points(results, frame);
      Mask d(frame, points);
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
