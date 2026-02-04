#pragma once
#ifdef USE_ORT

#include "legged_rl_deploy/policy/i_policy_runner.h"
#include <onnxruntime_cxx_api.h>

namespace legged_rl_deploy {

class OrtPolicyRunner final : public IPolicyRunner {
public:
  void load(const std::string& model_path, int input_dim, int output_dim) override;
  void infer(const float* input, float* output) override;

private:
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "legged_rl_deploy_ort"};
  Ort::SessionOptions opt_;
  std::unique_ptr<Ort::Session> session_;

  Ort::MemoryInfo mem_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

  std::string in_name_, out_name_;
  const char* in_name_c_{nullptr};
  const char* out_name_c_{nullptr};

  std::vector<int64_t> in_shape_{1, 0};   // [1, input_dim]
  int64_t out_dim64_{0};

  std::vector<float> in_buf_;             // size = input_dim
};

}  // namespace legged_rl_deploy

#endif
