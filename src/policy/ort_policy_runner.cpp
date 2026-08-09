#include "legged_rl_deploy/policy/ort_policy_runner.h"

#ifdef USE_ORT

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace legged_rl_deploy {
namespace {

void validateTensor(const Ort::TypeInfo& type_info,
                    const IPolicyRunner::TensorSpec& expected,
                    const std::string& context) {
  const auto info = type_info.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(context + " must be float32");
  }
  const auto actual_shape = info.GetShape();
  const bool shape_matches =
      actual_shape.size() == expected.shape.size() &&
      std::equal(actual_shape.begin(), actual_shape.end(), expected.shape.begin(),
                 [](int64_t actual, int64_t configured) {
                   return actual == configured || actual == -1;
                 });
  if (!shape_matches) {
    throw std::runtime_error(context + " shape does not match config");
  }
}

}  // namespace

void OrtPolicyRunner::loadBackend(const std::string& model_path) {
  options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
  options_.SetIntraOpNumThreads(1);
  session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options_);

  const auto& expected_inputs = modelInputs();
  const auto& expected_outputs = modelOutputs();
  if (session_->GetInputCount() != expected_inputs.size() ||
      session_->GetOutputCount() != expected_outputs.size()) {
    throw std::runtime_error("ORT model input/output count does not match config");
  }

  Ort::AllocatorWithDefaultOptions allocator;
  input_names_.clear();
  for (size_t i = 0; i < expected_inputs.size(); ++i) {
    auto allocated = session_->GetInputNameAllocated(i, allocator);
    const std::string actual_name = allocated.get();
    if (!expected_inputs[i].name.empty() && actual_name != expected_inputs[i].name) {
      throw std::runtime_error("ORT input " + std::to_string(i) + " is named " +
                               actual_name + ", expected " + expected_inputs[i].name);
    }
    validateTensor(session_->GetInputTypeInfo(i), expected_inputs[i],
                   "ORT input " + actual_name);
    input_names_.push_back(actual_name);
  }

  output_names_.clear();
  for (size_t i = 0; i < expected_outputs.size(); ++i) {
    auto allocated = session_->GetOutputNameAllocated(i, allocator);
    const std::string actual_name = allocated.get();
    if (!expected_outputs[i].name.empty() && actual_name != expected_outputs[i].name) {
      throw std::runtime_error("ORT output " + std::to_string(i) + " is named " +
                               actual_name + ", expected " + expected_outputs[i].name);
    }
    validateTensor(session_->GetOutputTypeInfo(i), expected_outputs[i],
                   "ORT output " + actual_name);
    output_names_.push_back(actual_name);
  }

  input_name_ptrs_.clear();
  for (const auto& name : input_names_) input_name_ptrs_.push_back(name.c_str());
  output_name_ptrs_.clear();
  for (const auto& name : output_names_) output_name_ptrs_.push_back(name.c_str());
}

void OrtPolicyRunner::runBackend(const std::vector<const float*>& inputs,
                                 const std::vector<float*>& outputs) {
  if (!session_) throw std::runtime_error("ORT session is not loaded");
  const auto& input_specs = modelInputs();
  const auto& output_specs = modelOutputs();

  std::vector<Ort::Value> tensors;
  tensors.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_, const_cast<float*>(inputs[i]), input_specs[i].size,
        input_specs[i].shape.data(), input_specs[i].shape.size()));
  }

  auto results = session_->Run(Ort::RunOptions{nullptr}, input_name_ptrs_.data(),
                               tensors.data(), tensors.size(),
                               output_name_ptrs_.data(), output_name_ptrs_.size());
  if (results.size() != outputs.size()) {
    throw std::runtime_error("ORT returned an unexpected output count");
  }
  for (size_t i = 0; i < results.size(); ++i) {
    const float* data = results[i].GetTensorData<float>();
    std::copy(data, data + output_specs[i].size, outputs[i]);
  }
}

}  // namespace legged_rl_deploy

#endif
