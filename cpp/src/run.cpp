// main entry point
#include <mlsamples/pipeline.h>
//
#include <argparse/argparse.hpp>

//
#include <exception>
#include <filesystem>

namespace mlsamples {
void run_task(Backend b, Task t,
              std::filesystem::path in_video,
              std::filesystem::path out) {
  Pipeline pipe(t, b, out);
  pipe.run(in_video);
}

} // namespace mlsamples

int main(int argc, char *argv[]) {
  //
  argparse::ArgumentParser parser("ai task runner");
  parser.add_argument("--backend")
      .default_value("yolo")
      .choices("yolo");
  parser.add_argument("--task")
      .default_value("segment")
      .choices("segment", "detect", "pose")
      .required();
  parser.add_argument("--video")
      .help("path to input video")
      .required();

  parser.add_argument("--save_name")
      .help("save name for output video, e.g. out.mp4")
      .default_value("out.mp4")
      .required();
  try {
    parser.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << parser;
    std::exit(1);
  }
  std::string msg = "unexpected argument for ";
  //
  std::string back = parser.get<std::string>("--backend");
  if (back != std::string("yolo")) {
    std::string m = msg + "--backend ";
    m += back;
    throw std::runtime_error(m.c_str());
  }
  mlsamples::Backend b = mlsamples::Backend::YOLO;
  //
  mlsamples::Task t = mlsamples::Task::SEGMENTATION;
  std::string p_t = parser.get<std::string>("--task");
  if (p_t == std::string("segment")) {
    t = mlsamples::Task::SEGMENTATION;
  } else if (p_t == std::string("detect")) {
    t = mlsamples::Task::DETECTION;
  } else if (p_t == std::string("pose")) {
    t = mlsamples::Task::POSE_ESTIMATION;
  } else {
    std::string m = msg + "--task ";
    m += p_t;
    throw std::runtime_error(m);
  }
  //
  std::string in_v = parser.get<std::string>("--video");
  std::filesystem::path in_video(in_v);
  in_video =
      std::filesystem::absolute(in_video).make_preferred();
  if (!std::filesystem::exists(in_video)) {
    std::string m("argument to --video ");
    m += in_v;
    m += "  doesn't exist";
    throw std::runtime_error(m.c_str());
  }

  std::string s_v = parser.get<std::string>("--save_name");
  std::filesystem::path out_v(s_v);
  out_v = std::filesystem::absolute(out_v).make_preferred();
  mlsamples::run_task(b, t, in_video, out_v);
  std::cout << "all done" << std::endl;
  return 0;
}
