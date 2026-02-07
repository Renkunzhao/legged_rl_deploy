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

  // --- all model inputs (enumerated at load time) ---
  std::vector<std::string> in_names_;
  std::vector<const char*> in_names_c_;

  size_t obs_idx_{0};
  std::vector<int64_t> obs_shape_{1, 0};
  std::vector<float> obs_buf_;

  // Zero-filled buffers for extra inputs (e.g. time_step) — ignored
  std::vector<std::vector<float>> extra_bufs_;
  std::vector<std::vector<int64_t>> extra_shapes_;

  // --- all model outputs ---
  std::vector<std::string> out_names_;
  std::vector<const char*> out_names_c_;
  size_t action_idx_{0};
  int64_t out_dim64_{0};
};

}  // namespace legged_rl_deploy

#endif
