#include "drawer.h"

// other stuff
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

namespace mlsamples {
//
std::vector<cv::Mat>
draw(const std::vector<Detection> &ds) {
  //
  std::vector<cv::Mat> results;
  for (auto d : ds) {
    torch::Tensor frame_t = d.frame.to(torch::kU8);
    auto frame_sizes = frame_t.sizes();
    int width = frame_sizes[1];
    int height = frame_sizes[0];
    cv::Mat temp(width, height, CV_8UC3,
                 frame_t.data_ptr());
    cv::Mat frame = temp.clone();
    torch::Tensor bboxes = d.bboxes;
    auto sizes = bboxes.sizes();
    int nb_boxes = static_cast<int>(sizes[0]);
    for (int i = 0; i < nb_boxes; ++i) {
      auto box =
          bboxes.index({i, torch::Slice(NULL)}).flatten();
      auto x1 = static_cast<int>(box[0]);
      auto y1 = static_cast<int>(box[1]);
      auto w = static_cast<int>(box[2]);
      auto h = static_cast<int>(box[3]);
      x2 = x1 + w;
      y2 = y1 + h;
      cv::Point p1(x1, y1);
      cv::Point p2(x2, y2);

      cv::rectangle(frame, p1, p2, cv::Scalar(0));
    }
    results.push_back(frame);
  }
  return results;
}
} // namespace mlsamples
