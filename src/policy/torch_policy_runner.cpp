#include "legged_rl_deploy/policy/torch_policy_runner.h"

#ifdef USE_TORCH
#include <cstring>
#include <stdexcept>

namespace legged_rl_deploy {

static torch::Tensor extractActionTensor(const torch::jit::IValue& out_iv) {
  if (out_iv.isTensor()) return out_iv.toTensor();
  if (out_iv.isTuple()) {
    const auto& elems = out_iv.toTuple()->elements();
    if (!elems.empty() && elems[0].isTensor()) return elems[0].toTensor();
  }
  throw std::runtime_error("Policy output must be Tensor or tuple[0]=Tensor");
}

void TorchPolicyRunner::load(const std::string& model_path, int input_dim, int output_dim) {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  policy_ = torch::jit::load(model_path);
  policy_.eval();

  // 预分配 input tensor（CPU float32）
  input_ = torch::empty({1, input_dim_}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  torch::NoGradGuard ng;
  auto out_iv = policy_.forward({input_});
  auto out = extractActionTensor(out_iv).to(torch::kCPU, torch::kFloat32).contiguous();
  if (out.dim() != 2 || out.size(0) != 1 || out.size(1) != output_dim_) {
    throw std::runtime_error("Policy output shape mismatch in self-check");
  }
}

void TorchPolicyRunner::infer(const float* input, float* output) {
  // 把 obs 写入预分配 tensor（避免每次创建 tensor）
  std::memcpy(input_.data_ptr<float>(), input, sizeof(float) * input_dim_);

  torch::NoGradGuard ng;
  auto out_iv = policy_.forward({input_});
  auto out = extractActionTensor(out_iv).to(torch::kCPU, torch::kFloat32).contiguous();

  // 这里可以选择不每次检查 shape（release 模式略过），debug 模式可保留 assert
  std::memcpy(output, out.data_ptr<float>(), sizeof(float) * output_dim_);
}

}  // namespace legged_rl_deploy

#endif
