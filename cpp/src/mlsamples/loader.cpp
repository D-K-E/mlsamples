#include <mlsamples/loader.h>
#include <torch/script.h> // One-stop header.

namespace mlsamples {

torch::jit::Module load(std::filesystem::path model_p) {

  torch::jit::script::Module m;
  try {
    std::string s = model_p.str();
    m = torch::jit::load(s);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model:" << std::endl;
    throw std::runtime_error(e.what());
  }
  return m;
}
} // namespace mlsamples
