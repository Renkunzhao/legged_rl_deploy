#include "legged_rl_deploy/policy/ort_policy_runner.h"

#ifdef USE_ORT
#include <cstring>
#include <stdexcept>

namespace legged_rl_deploy {

static inline int64_t numel(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (auto v : s) n *= (v > 0 ? v : 1);
  return n;
}

void OrtPolicyRunner::load(const std::string& model_path, int input_dim, int output_dim) {
  input_dim_  = input_dim;
  output_dim_ = output_dim;
  out_dim64_  = static_cast<int64_t>(output_dim_);

  opt_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  opt_.SetIntraOpNumThreads(1);

  session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opt_);

  // 取 input/output 名字（假设各 1 个）
  if (session_->GetInputCount() < 1 || session_->GetOutputCount() < 1) {
    throw std::runtime_error("ORT model must have at least 1 input and 1 output");
  }

  Ort::AllocatorWithDefaultOptions alloc;
  {
    auto n = session_->GetInputNameAllocated(0, alloc);
    in_name_ = n.get();
    in_name_c_ = in_name_.c_str();
  }
  {
    auto n = session_->GetOutputNameAllocated(0, alloc);
    out_name_ = n.get();
    out_name_c_ = out_name_.c_str();
  }

  // 预分配 input buffer
  in_shape_[0] = 1;
  in_shape_[1] = static_cast<int64_t>(input_dim_);
  in_buf_.assign(static_cast<size_t>(input_dim_), 0.0f);

  // self-check：跑一次，检查输出维度是否匹配
  Ort::Value in_t = Ort::Value::CreateTensor<float>(
      mem_, in_buf_.data(), in_buf_.size(), in_shape_.data(), in_shape_.size());

  auto outs = session_->Run(
      Ort::RunOptions{nullptr},
      &in_name_c_, &in_t, 1,
      &out_name_c_, 1);

  if (outs.size() != 1) throw std::runtime_error("ORT output count mismatch");

  auto info  = outs[0].GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();  // e.g. [1, out_dim] or [out_dim]

  int64_t n = 0;
  if (shape.size() == 2) n = shape[0] * shape[1];
  else if (shape.size() == 1) n = shape[0];
  else n = numel(shape);

  if (n != out_dim64_) {
    throw std::runtime_error("ORT policy output size mismatch in self-check");
  }
}

void OrtPolicyRunner::infer(const float* input, float* output) {
  if (!session_) throw std::runtime_error("ORT session not loaded");

  std::memcpy(in_buf_.data(), input, sizeof(float) * static_cast<size_t>(input_dim_));

  Ort::Value in_t = Ort::Value::CreateTensor<float>(
      mem_, in_buf_.data(), in_buf_.size(), in_shape_.data(), in_shape_.size());

  auto outs = session_->Run(
      Ort::RunOptions{nullptr},
      &in_name_c_, &in_t, 1,
      &out_name_c_, 1);

  float* out_ptr = outs[0].GetTensorMutableData<float>();
  std::memcpy(output, out_ptr, sizeof(float) * static_cast<size_t>(output_dim_));
}

}  // namespace legged_rl_deploy

#endif
