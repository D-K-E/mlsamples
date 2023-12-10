#ifndef PIPELINE_H
#define PIPELINE_H

#include <memory>
namespace mlsamples {

enum class Task {
  SEGMENTATION = 1,
  DETECTION = 2,
  POSE_ESTIMATION = 3,
};
enum class Backend { YOLO = 1 };

class Pipeline {
public:
  Pipeline(Backend b, Task t, std::filesystem::path out_p);
  ~Pipeline();
  void run(std::filesystem::path in_video);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace mlsamples
#endif
