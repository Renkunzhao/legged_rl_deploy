#include "legged_rl_deploy/policy/torch_policy_runner.h"

#ifdef USE_TORCH

#include <algorithm>
#include <stdexcept>

namespace legged_rl_deploy {
namespace {

void copyTensor(const torch::jit::IValue& value,
                const IPolicyRunner::TensorSpec& expected, float* output) {
  if (!value.isTensor()) throw std::runtime_error("Torch policy output must be Tensor");
  torch::Tensor tensor =
      value.toTensor().to(torch::kCPU, torch::kFloat32).contiguous();
  if (tensor.sizes().vec() != expected.shape) {
    throw std::runtime_error("Torch policy output shape does not match config");
  }
  std::copy(tensor.data_ptr<float>(), tensor.data_ptr<float>() + expected.size, output);
}

}  // namespace

void TorchPolicyRunner::loadBackend(const std::string& model_path) {
  policy_ = torch::jit::load(model_path);
  policy_.eval();

  const auto schema = policy_.get_method("forward").function().getSchema();
  const auto& arguments = schema.arguments();
  if (arguments.size() != modelInputs().size() + 1) {
    throw std::runtime_error("Torch forward input count does not match config");
  }
  for (size_t i = 0; i < modelInputs().size(); ++i) {
    if (!modelInputs()[i].name.empty() &&
        arguments[i + 1].name() != modelInputs()[i].name) {
      throw std::runtime_error("Torch forward input name does not match config: " +
                               modelInputs()[i].name);
    }
  }
}

void TorchPolicyRunner::runBackend(const std::vector<const float*>& inputs,
                                   const std::vector<float*>& outputs) {
  std::vector<torch::jit::IValue> arguments;
  arguments.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    arguments.emplace_back(torch::from_blob(
        const_cast<float*>(inputs[i]), modelInputs()[i].shape,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)));
  }

  torch::NoGradGuard no_grad;
  const torch::jit::IValue result = policy_.forward(arguments);
  if (modelOutputs().size() == 1 && result.isTensor()) {
    copyTensor(result, modelOutputs()[0], outputs[0]);
    return;
  }
  if (!result.isTuple()) {
    throw std::runtime_error("Torch multi-output policy must return a tuple");
  }
  const auto& values = result.toTuple()->elements();
  if (values.size() != modelOutputs().size()) {
    throw std::runtime_error("Torch forward output count does not match config");
  }
  for (size_t i = 0; i < values.size(); ++i) {
    copyTensor(values[i], modelOutputs()[i], outputs[i]);
  }
}

void TorchPolicyRunner::resetBackend() {
  if (!policy_.find_method("reset")) return;
  const auto method = policy_.get_method("reset");
  if (method.function().getSchema().arguments().size() != 1) {
    throw std::runtime_error("Torch reset() must not take arguments");
  }
  torch::NoGradGuard no_grad;
  method({});
}

}  // namespace legged_rl_deploy

#endif
