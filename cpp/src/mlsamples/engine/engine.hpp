#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <mlsamples/pipeline.h>
#include <mlsamples/task/detection/interface.h>
#include <vector>

namespace mlsamples {

template <class TaskHandler, class TaskOutput>
class Engine {
public:
  Engine(std::unique_ptr<TaskHandler> d)
      : handler(std::move(d)) {}

  std::vector<TaskOutput> run(std::filesystem::path video) {
    auto ds = handler->run(video);
    return ds;
  }

private:
  std::unique_ptr<TaskHandler> handler;
}

} // namespace mlsamples
#endif
