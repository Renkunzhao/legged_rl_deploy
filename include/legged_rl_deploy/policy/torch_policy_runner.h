#pragma once
#ifdef USE_TORCH

#include "legged_rl_deploy/policy/i_policy_runner.h"
#include <torch/script.h>

namespace legged_rl_deploy {

class TorchPolicyRunner final : public IPolicyRunner {
public:
  void load(const std::string& model_path, int input_dim, int output_dim) override;
  void infer(const float* input, float* output) override;

private:
  torch::jit::script::Module policy_;
  torch::Tensor input_;   // [1, input_dim] float32 CPU
};

}  // namespace legged_rl_deploy

#endif
