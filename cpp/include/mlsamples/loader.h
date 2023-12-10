//! \brief mlsamples torchscript model loader

#ifndef LOADER_H
#define LOADER_H

#include <filesystem>
#include <torch/torch.h>

namespace mlsamples {
torch::jit::Module load(std::filesystem::path model_p);
}

#endif
