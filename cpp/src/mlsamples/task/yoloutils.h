// simple utilities for yolo based funcs
#ifndef YOLOUTILS_H
#define YOLOUTILS_H
#include <mlsamples/pipeline.h>
#include <opencv2/core.hpp>
#include <torch/script.h>
#include <torch/torch.h>
namespace mlsamples {

const int kYOLO_NETWORK_WIDTH_ = 640;
const int kYOLO_NETWORK_HEIGHT_ = 640;
const int kYOLO_CHANNEL_ = 3;
const float kYOLO_CONFIDENCE_THRESHOLD = 0.35f;
const float kYOLO_NMS_THRESHOLD = 0.4f;

// from
// https://github.com/olibartfast/object-detection-inference/blob/master/src/libtorch/YoloV8.cpp
std::vector<float>
yolo_preprocess_image(const cv::Mat &image);

// from
// https://github.com/olibartfast/object-detection-inference/blob/master/src/libtorch/YoloV8.cpp
cv::Rect yolo_get_rect(const cv::Size &imgSz,
                       const std::vector<float> &bbox);

namespace detection {
std::vector<cv::Rect> yolo_postprocess(
    const std::vector<std::vector<float>> &outputs,
    const std::vector<std::vector<int64_t>> &shapes,
    const cv::Size &frame_size);
}

namespace segmentation {
struct SegOutput {
  SegOutput(const std::vector<cv::Rect> &bs,
            const cv::Mat &s, const cv::Mat &p,
            const cv::Rect &r
            );
  std::vector<cv::Rect> bboxes;
  cv::Mat segm;
  cv::Mat mask_proposals;
  cv::Rect roi;
};

SegOutput yolo_postprocess(
    const std::vector<std::vector<float>> &outputs,
    const std::vector<std::vector<int64_t>> &shapes,
    const cv::Size &frame_size);
} // namespace segmentation

struct YoloV8Model {
  YoloV8Model(Task t);

  std::pair<std::vector<std::vector<float>>,
            std::vector<std::vector<int64_t>>>
  infer(std::vector<torch::jit::IValue> inputs);

  torch::jit::Module model;
};

} // namespace mlsamples
#endif
