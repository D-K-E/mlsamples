// include related
// #include "engine/engine.hpp"
#include "visual/drawer.h"
//

#include <mlsamples/pipeline.h>

#include <mlsamples/task/detection/interface.h>
#include <mlsamples/task/detection/yolo.h>

#include <opencv2/videoio.hpp>

namespace mlsamples {
//
struct Pipeline::Impl {

  Impl(Task t, Backend b, std::filesystem::path o)
      : task(t), backend(b), save_loc(o) {
    if (t == Task::DETECTION) {
      if (backend == Backend::YOLO) {
        detector.reset(new detection::Yolo());
      }
    }
  }
  ~Impl() = default;
  void run(std::filesystem::path video) {
    std::vector<cv::Mat> frames;
    if (task == Task::DETECTION) {
      std::vector<detection::Detection> ds =
          detector->run(video);
      frames = draw(ds);
    }
    std::string filename = save_loc.string();
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    int fps = 30;
    cv::VideoWriter writer(filename, codec, fps,
                           frames[0].size(), true);
    for (auto frame : frames) {
      writer.write(frame);
    }
  }
  //
  Task task;
  Backend backend;
  std::filesystem::path save_loc;

  std::unique_ptr<detection::Detector> detector{nullptr};
};

Pipeline::Pipeline(Task t, Backend b,
                   std::filesystem::path o) {
  auto temp = std::make_unique<Impl>(t, b, o);
  impl = std::move(temp);
}
Pipeline::~Pipeline() = default;

void Pipeline::run(std::filesystem::path in_video) {
  impl->run(in_video);
}

} // namespace mlsamples
