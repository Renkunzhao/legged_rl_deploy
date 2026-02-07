#pragma once
#ifdef USE_ORT

#include "legged_rl_deploy/policy/i_policy_runner.h"
#include <onnxruntime_cxx_api.h>
#include <unordered_map>

namespace legged_rl_deploy {

class OrtPolicyRunner final : public IPolicyRunner {
public:
  void load(const std::string& model_path, int input_dim, int output_dim) override;
  void infer(const float* input, float* output) override;
  void setExtraInput(const std::string& name,
                     const float* data,
                     const std::vector<int64_t>& shape) override;

private:
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "legged_rl_deploy_ort"};
  Ort::SessionOptions opt_;
  std::unique_ptr<Ort::Session> session_;

  Ort::MemoryInfo mem_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

  // --- all model inputs (enumerated at load time) ---
  std::vector<std::string> in_names_;       // owned strings
  std::vector<const char*> in_names_c_;     // c_str pointers into in_names_

  // Primary obs input (index 0 in in_names_)
  size_t obs_idx_{0};                       // index of the "obs" input
  std::vector<int64_t> obs_shape_{1, 0};    // [1, input_dim]
  std::vector<float> obs_buf_;              // size = input_dim

  // Extra named inputs (e.g. time_step)
  struct ExtraInput {
    const float* data{nullptr};
    std::vector<int64_t> shape;
  };
  std::unordered_map<std::string, ExtraInput> extras_;
  // Scratch buffers for extra inputs that need a default value
  std::unordered_map<std::string, std::vector<float>> extra_defaults_;

  // --- all model outputs ---
  std::vector<std::string> out_names_;
  std::vector<const char*> out_names_c_;
  size_t action_idx_{0};                    // index of the first output (actions)
  int64_t out_dim64_{0};
};

}  // namespace legged_rl_deploy

#endif
