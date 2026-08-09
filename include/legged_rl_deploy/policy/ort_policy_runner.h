#pragma once
#ifdef USE_ORT

#include "legged_rl_deploy/policy/i_policy_runner.h"

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace legged_rl_deploy {

class OrtPolicyRunner final : public IPolicyRunner {
private:
  void loadBackend(const std::string& model_path) override;
  void runBackend(const std::vector<const float*>& inputs,
                  const std::vector<float*>& outputs) override;

  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "legged_rl_deploy_ort"};
  Ort::SessionOptions options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_{
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::vector<std::string> input_names_;
  std::vector<const char*> input_name_ptrs_;
  std::vector<std::string> output_names_;
  std::vector<const char*> output_name_ptrs_;
};

}  // namespace legged_rl_deploy

#endif
