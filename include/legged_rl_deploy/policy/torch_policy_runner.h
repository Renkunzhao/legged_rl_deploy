#pragma once
#ifdef USE_TORCH

#include "legged_rl_deploy/policy/i_policy_runner.h"

#include <torch/script.h>

namespace legged_rl_deploy {

class TorchPolicyRunner final : public IPolicyRunner {
private:
  void loadBackend(const std::string& model_path) override;
  void runBackend(const std::vector<const float*>& inputs,
                  const std::vector<float*>& outputs) override;
  void resetBackend() override;

  torch::jit::script::Module policy_;
};

}  // namespace legged_rl_deploy

#endif
