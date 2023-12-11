#include "yoloutils.h"
#include <mlsamples/loader.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
// implementation file

namespace mlsamples {
//
std::vector<float>
yolo_preprocess_image(const cv::Mat &image) {
  cv::Mat blob;
  cv::cvtColor(image, blob, cv::COLOR_BGR2RGB);
  int target_width, target_height, offset_x, offset_y;
  float resize_ratio_width =
      static_cast<float>(kYOLO_NETWORK_WIDTH_) /
      static_cast<float>(image.cols);
  float resize_ratio_height =
      static_cast<float>(kYOLO_NETWORK_HEIGHT_) /
      static_cast<float>(image.rows);

  if (resize_ratio_height > resize_ratio_width) {
    target_width = kYOLO_NETWORK_WIDTH_;
    target_height = resize_ratio_width * image.rows;
    offset_x = 0;
    offset_y = (kYOLO_NETWORK_HEIGHT_ - target_height) / 2;
  } else {
    target_width = resize_ratio_height * image.cols;
    target_height = kYOLO_NETWORK_HEIGHT_;
    offset_x = (kYOLO_NETWORK_WIDTH_ - target_width) / 2;
    offset_y = 0;
  }

  cv::Mat resized_image(target_height, target_width,
                        CV_8UC3);
  cv::resize(blob, resized_image, resized_image.size(), 0,
             0, cv::INTER_LINEAR);
  cv::Mat output_image(kYOLO_NETWORK_WIDTH_,
                       kYOLO_NETWORK_HEIGHT_, CV_8UC3,
                       cv::Scalar(128, 128, 128));
  resized_image.copyTo(output_image(
      cv::Rect(offset_x, offset_y, resized_image.cols,
               resized_image.rows)));
  output_image.convertTo(output_image, CV_32FC3,
                         1.f / 255.f);

  size_t img_byte_size =
      output_image.total() *
      output_image.elemSize(); // Allocate a buffer to hold
                               // all image elements.
  std::vector<float> input_data = std::vector<float>(
      kYOLO_NETWORK_WIDTH_ * kYOLO_NETWORK_HEIGHT_ *
      kYOLO_CHANNEL_);
  std::memcpy(input_data.data(), output_image.data,
              img_byte_size);

  std::vector<cv::Mat> chw;
  for (size_t i = 0; i < kYOLO_CHANNEL_; ++i) {
    chw.emplace_back(
        cv::Mat(cv::Size(kYOLO_NETWORK_WIDTH_,
                         kYOLO_NETWORK_HEIGHT_),
                CV_32FC1,
                &(input_data[i * kYOLO_NETWORK_WIDTH_ *
                             kYOLO_NETWORK_HEIGHT_])));
  }
  cv::split(output_image, chw);

  return input_data;
}

cv::Rect yolo_get_rect(const cv::Size &imgSz,
                       const std::vector<float> &bbox) {
  float r_w = kYOLO_NETWORK_WIDTH_ /
              static_cast<float>(imgSz.width);
  float r_h = kYOLO_NETWORK_HEIGHT_ /
              static_cast<float>(imgSz.height);

  int l, r, t, b;
  if (r_h > r_w) {
    l = bbox[0] - bbox[2] / 2.f;
    r = bbox[0] + bbox[2] / 2.f;
    t = bbox[1] - bbox[3] / 2.f -
        (kYOLO_NETWORK_HEIGHT_ - r_w * imgSz.height) / 2;
    b = bbox[1] + bbox[3] / 2.f -
        (kYOLO_NETWORK_HEIGHT_ - r_w * imgSz.height) / 2;
    l /= r_w;
    r /= r_w;
    t /= r_w;
    b /= r_w;
  } else {
    l = bbox[0] - bbox[2] / 2.f -
        (kYOLO_NETWORK_WIDTH_ - r_h * imgSz.width) / 2;
    r = bbox[0] + bbox[2] / 2.f -
        (kYOLO_NETWORK_WIDTH_ - r_h * imgSz.width) / 2;
    t = bbox[1] - bbox[3] / 2.f;
    b = bbox[1] + bbox[3] / 2.f;
    l /= r_h;
    r /= r_h;
    t /= r_h;
    b /= r_h;
  }

  // Clamp the coordinates within the image bounds
  l = std::max(0, std::min(l, imgSz.width - 1));
  r = std::max(0, std::min(r, imgSz.width - 1));
  t = std::max(0, std::min(t, imgSz.height - 1));
  b = std::max(0, std::min(b, imgSz.height - 1));

  return cv::Rect(l, t, r - l, b - t);
}

namespace detection {
std::vector<cv::Rect> yolo_postprocess(
    const std::vector<std::vector<float>> &outputs,
    const std::vector<std::vector<int64_t>> &shapes,
    const cv::Size &frame_size) {
  const float *output0 = outputs.front().data();
  const std::vector<int64_t> shape0 = shapes.front();

  const auto offset = 4;
  const auto num_classes = shape0[1] - offset;

  std::vector<float> out_d(outputs[0].begin(),
                           outputs[0].end());
  cv::Mat outm(shape0[1], shape0[2], CV_32FC1,
               out_d.data());
  cv::Mat tout_m = outm.t();

  std::vector<cv::Rect> boxes;
  std::vector<float> confs;

  // Get all the YOLO proposals
  for (int i = 0; i < shape0[2]; ++i) {
    cv::Mat trow = tout_m.row(i);
    cv::Mat scores_t = trow.colRange(4, trow.cols);
    cv::Mat box_m = trow.colRange(0, 4);
    auto score_it = std::max_element(
        scores_t.begin<float>(), scores_t.end<float>());
    float m_score = *score_it;
    if (m_score > kYOLO_CONFIDENCE_THRESHOLD) {
      std::vector<float> bdata(box_m.begin<float>(),
                               box_m.end<float>());
      cv::Rect y_rect = yolo_get_rect(frame_size, bdata);
      boxes.emplace_back(y_rect);
      confs.emplace_back(m_score);
    }
  }

  // Perform Non Maximum Suppression and draw predictions.
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confs,
                    kYOLO_CONFIDENCE_THRESHOLD,
                    kYOLO_NMS_THRESHOLD, indices);
  std::vector<cv::Rect> rects;
  for (int i = 0; i < indices.size(); i++) {
    int idx = indices[i];
    cv::Rect box = boxes[idx];
    rects.push_back(box);
  }
  return rects;
}
} // namespace detection

namespace segmentation {

// taken from
// https://github.com/olibartfast/object-detection-inference/blob/master/src/libtorch/YoloV8.cpp
cv::Rect find_pad_size(const size_t inputW,
                       const size_t inputH,
                       const cv::Size &inputSize) {
  std::vector<int> padSize;
  float w, h, x, y;
  float r_w = inputW / (inputSize.width * 1.0);
  float r_h = inputH / (inputSize.height * 1.0);
  if (r_h > r_w) {
    w = inputW;
    h = r_w * inputSize.height;
    x = 0;
    y = (inputH - h) / 2;
  } else {
    w = r_h * inputSize.width;
    h = inputH;
    x = (inputW - w) / 2;
    y = 0;
  }
  return cv::Rect(x, y, w, h);
}

SegOutput::SegOutput(const std::vector<cv::Rect> &bs,
                     const cv::Mat &s, const cv::Mat &p,
                     const cv::Rect &r)
    : bboxes(bs), segm(s), mask_proposals(p), roi(r) {}

SegOutput yolo_postprocess(
    const std::vector<std::vector<float>> &output,
    const std::vector<std::vector<int64_t>> &shape,
    const cv::Size &frame_size) {
  const auto offset = 4;
  const auto num_classes =
      shape[0][1] - offset - shape[1][1];

  // Construct output matrix
  std::vector<float> mat_d(output[0].size());
  std::copy(output[0].begin(), output[0].end(),
            mat_d.begin());
  cv::Mat outm(shape[0][1], shape[0][2], CV_32FC1,
               mat_d.data());

  // Transpose output matrix
  cv::Mat tout_m = outm.t();

  std::vector<cv::Rect> boxes;
  std::vector<float> confs;

  std::vector<std::vector<float>> picked_proposals;

  // Get all the YOLO proposals
  for (int i = 0; i < shape[0][2]; ++i) {
    cv::Mat row = tout_m.row(i);
    cv::Mat scores = row.colRange(4, num_classes);
    auto it = std::max_element(scores.begin<float>(),
                               scores.end<float>());
    float score = *it;
    if (score > kYOLO_CONFIDENCE_THRESHOLD) {
      cv::Mat mbox = row.colRange(0, 4);
      std::vector<float> mm(mbox.begin<float>(),
                            mbox.end<float>());
      boxes.emplace_back(yolo_get_rect(frame_size, mm));
      confs.emplace_back(score);
      int end = num_classes + shape[1][1];
      cv::Mat prange = row.colRange(num_classes, end);
      std::vector<float> proposal(prange.begin<float>(),
                                  prange.end<float>());
      picked_proposals.emplace_back(proposal);
    }
  }

  // Perform Non Maximum Suppression and draw predictions.
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confs,
                    kYOLO_CONFIDENCE_THRESHOLD,
                    kYOLO_NMS_THRESHOLD, indices);
  int sc, sh, sw;
  std::tie(sc, sh, sw) =
      std::make_tuple(static_cast<int>(shape[1][1]),
                      static_cast<int>(shape[1][2]),
                      static_cast<int>(shape[1][3]));

  std::vector<float> mask_d(output[1].begin(),
                            output[1].end());
  cv::Mat segm =
      cv::Mat(sc, sw * sh, CV_32FC1, mask_d.data());
  cv::Rect pad_rect =
      find_pad_size(kYOLO_NETWORK_WIDTH_,
                    kYOLO_NETWORK_HEIGHT_, frame_size);
  cv::Rect roi(
      int((float)pad_rect.x / kYOLO_NETWORK_WIDTH_ * sw),
      int((float)pad_rect.y / kYOLO_NETWORK_HEIGHT_ * sh),
      int(sw - pad_rect.x / 2), int(sh - pad_rect.y / 2));

  cv::Mat maskProposals;
  std::vector<cv::Rect> bboxes;
  for (int i = 0; i < indices.size(); i++) {
    int idx = indices[i];
    std::vector<float> picked = picked_proposals[idx];
    cv::Rect bbox = boxes[idx];
    bboxes.push_back(bbox);
    cv::Mat pmat = cv::Mat(picked).t();
    maskProposals.push_back(pmat);
  }
  SegOutput seg(bboxes, segm, maskProposals, roi);

  return seg;
}

} // namespace segmentation

YoloV8Model::YoloV8Model(Task t) {
  std::filesystem::path p(ASSET_DIR);
  p = p.make_preferred();
  if (t == Task::DETECTION) {
    p /= std::filesystem::path("yolov8x.torchscript");
  } else if (t == Task::SEGMENTATION) {
    p /= std::filesystem::path("yolov8x-seg.torchscript");
    // p /= std::filesystem::path("yolov8x.torchscript");
  } else {
    throw std::runtime_error("Unsupported task");
  }

  if (!std::filesystem::exists(p)) {
    std::string m("model can not "
                  "be found in: ");
    m += p.string();
    throw std::runtime_error(m.c_str());
  }
  model = mlsamples::load(p);
  model.eval();
}
std::pair<std::vector<std::vector<float>>,
          std::vector<std::vector<int64_t>>>
YoloV8Model::infer(std::vector<torch::jit::IValue> inputs) {
  // from
  // https://github.com/olibartfast/object-detection-inference/blob/master/src/libtorch/LibtorchInfer.cpp
  //
  auto output = model.forward(inputs);
  std::vector<std::vector<float>> output_vectors;
  std::vector<std::vector<int64_t>> shape_vectors;

  if (output.isTuple()) {
    // Handle the case where the model returns a tuple
    auto tuple_outputs = output.toTuple()->elements();

    for (const auto &output_tensor : tuple_outputs) {
      if (!output_tensor.isTensor())
        continue;
      torch::Tensor tensor = output_tensor.toTensor()
                                 .to(torch::kCPU)
                                 .contiguous();

      // Get the output data as a float pointer
      const float *output_data = tensor.data_ptr<float>();

      // Store the output data in the outputs vector
      std::vector<float> output_vector(
          output_data, output_data + tensor.numel());
      output_vectors.push_back(output_vector);

      // Get the shape of the output tensor
      std::vector<int64_t> shape = tensor.sizes().vec();
      shape_vectors.push_back(shape);
    }
  } else {
    torch::Tensor tensor = output.toTensor();
    if (tensor.size(0) == 1) {
      // If there's only one output tensor
      torch::Tensor output_tensor =
          tensor.to(torch::kCPU).contiguous();

      // Get the output data as a float pointer
      const float *output_data =
          output_tensor.data_ptr<float>();

      // Store the output data and shape in vectors
      output_vectors.emplace_back(
          output_data, output_data + output_tensor.numel());
      shape_vectors.push_back(output_tensor.sizes().vec());
    } else {
      for (int i = 0; i < tensor.size(0); ++i) {
        torch::Tensor output_tensor =
            tensor[i].to(torch::kCPU).contiguous();

        // Get the output data as a float pointer
        const float *output_data =
            output_tensor.data_ptr<float>();

        // Store the output data and shape in vectors
        output_vectors.emplace_back(
            output_data,
            output_data + output_tensor.numel());
        shape_vectors.push_back(
            output_tensor.sizes().vec());
      }
    }
  }
  auto p = std::make_pair(output_vectors, shape_vectors);
  return p;
}

} // namespace mlsamples
