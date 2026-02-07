#include "legged_rl_deploy/policy/ort_policy_runner.h"

#ifdef USE_ORT
#include <cstring>
#include <iostream>
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

  opt_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
  opt_.SetIntraOpNumThreads(1);

  session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opt_);

  if (session_->GetInputCount() < 1 || session_->GetOutputCount() < 1) {
    throw std::runtime_error("ORT model must have at least 1 input and 1 output");
  }

  Ort::AllocatorWithDefaultOptions alloc;

  // ---------- enumerate all inputs ----------
  const size_t n_in = session_->GetInputCount();
  in_names_.resize(n_in);
  in_names_c_.resize(n_in);

  for (size_t i = 0; i < n_in; ++i) {
    auto n = session_->GetInputNameAllocated(i, alloc);
    in_names_[i] = n.get();
    in_names_c_[i] = in_names_[i].c_str();
  }

  // Find the primary obs input (first input, or named "obs")
  obs_idx_ = 0;
  for (size_t i = 0; i < n_in; ++i) {
    if (in_names_[i] == "obs") { obs_idx_ = i; break; }
  }

  // Prepare obs buffer
  obs_shape_[0] = 1;
  obs_shape_[1] = static_cast<int64_t>(input_dim_);
  obs_buf_.assign(static_cast<size_t>(input_dim_), 0.0f);

  // Zero-fill buffers for extra inputs (e.g. time_step) — ignored
  extra_bufs_.resize(n_in);
  extra_shapes_.resize(n_in);
  for (size_t i = 0; i < n_in; ++i) {
    if (i == obs_idx_) continue;
    auto type_info = session_->GetInputTypeInfo(i);
    auto shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
    extra_bufs_[i].assign(static_cast<size_t>(numel(shape)), 0.0f);
    extra_shapes_[i] = shape;
    std::cout << "[OrtPolicyRunner] Extra input \"" << in_names_[i]
              << "\" (ignored, zero-filled)" << std::endl;
  }

  // ---------- enumerate all outputs ----------
  const size_t n_out = session_->GetOutputCount();
  out_names_.resize(n_out);
  out_names_c_.resize(n_out);

  for (size_t i = 0; i < n_out; ++i) {
    auto n = session_->GetOutputNameAllocated(i, alloc);
    out_names_[i] = n.get();
    out_names_c_[i] = out_names_[i].c_str();
  }

  // First output is the action tensor
  action_idx_ = 0;

  // ---------- self-check ----------
  std::vector<Ort::Value> input_tensors;
  input_tensors.reserve(n_in);
  for (size_t i = 0; i < n_in; ++i) {
    if (i == obs_idx_) {
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
          mem_, obs_buf_.data(), obs_buf_.size(),
          obs_shape_.data(), obs_shape_.size()));
    } else {
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
          mem_, extra_bufs_[i].data(), extra_bufs_[i].size(),
          extra_shapes_[i].data(), extra_shapes_[i].size()));
    }
  }

  auto outs = session_->Run(
      Ort::RunOptions{nullptr},
      in_names_c_.data(), input_tensors.data(), n_in,
      out_names_c_.data(), n_out);

  if (outs.empty()) throw std::runtime_error("ORT output count = 0");

  auto info  = outs[action_idx_].GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();

  int64_t n = 0;
  if (shape.size() == 2) n = shape[0] * shape[1];
  else if (shape.size() == 1) n = shape[0];
  else n = numel(shape);

  if (n != out_dim64_) {
    throw std::runtime_error("ORT policy output size mismatch in self-check: got " +
                             std::to_string(n) + " expected " + std::to_string(out_dim64_));
  }
}

void OrtPolicyRunner::infer(const float* input, float* output) {
  if (!session_) throw std::runtime_error("ORT session not loaded");

  std::memcpy(obs_buf_.data(), input, sizeof(float) * static_cast<size_t>(input_dim_));

  const size_t n_in = in_names_.size();
  const size_t n_out = out_names_.size();

  std::vector<Ort::Value> input_tensors;
  input_tensors.reserve(n_in);
  for (size_t i = 0; i < n_in; ++i) {
    if (i == obs_idx_) {
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
          mem_, obs_buf_.data(), obs_buf_.size(),
          obs_shape_.data(), obs_shape_.size()));
    } else {
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
          mem_, extra_bufs_[i].data(), extra_bufs_[i].size(),
          extra_shapes_[i].data(), extra_shapes_[i].size()));
    }
  }

  auto outs = session_->Run(
      Ort::RunOptions{nullptr},
      in_names_c_.data(), input_tensors.data(), n_in,
      out_names_c_.data(), n_out);

  float* out_ptr = outs[action_idx_].GetTensorMutableData<float>();
  std::memcpy(output, out_ptr, sizeof(float) * static_cast<size_t>(output_dim_));
}

}  // namespace legged_rl_deploy

#endif
