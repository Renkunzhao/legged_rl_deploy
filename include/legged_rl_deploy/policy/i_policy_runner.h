#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace legged_rl_deploy {

struct RuntimeTensor {
  std::string source;
  const float* data = nullptr;
  size_t size = 0;
};

struct RuntimeInputSpec {
  std::string source;
  std::vector<int64_t> shape;
  size_t size = 0;
};

class IPolicyRunner {
public:
  struct TensorSpec {
    std::string name;
    std::vector<int64_t> shape;
    size_t size = 0;
    std::string binding;
  };

  virtual ~IPolicyRunner() = default;

  void load(const std::string& model_path, const YAML::Node& policy_node);
  void infer(const std::vector<RuntimeTensor>& runtime_inputs, float* actions);
  void reset();

  size_t actionDim() const { return action_dim_; }
  size_t observationDim() const;
  const std::vector<RuntimeInputSpec>& runtimeInputSpecs() const {
    return runtime_input_specs_;
  }

protected:
  const std::vector<TensorSpec>& modelInputs() const { return model_inputs_; }
  const std::vector<TensorSpec>& modelOutputs() const { return model_outputs_; }

  virtual void loadBackend(const std::string& model_path) = 0;
  virtual void runBackend(const std::vector<const float*>& inputs,
                          const std::vector<float*>& outputs) = 0;
  virtual void resetBackend() {}

private:
  struct StateBuffer {
    std::vector<int64_t> shape;
    std::vector<float> current;
    std::vector<float> next;
    float max_norm = 0.0f;
  };

  void parseContract(const YAML::Node& policy_node);
  void run(const std::vector<RuntimeTensor>& runtime_inputs, float* actions);
  static size_t tensorSize(const std::vector<int64_t>& shape,
                           const std::string& context);

  std::vector<TensorSpec> model_inputs_;
  std::vector<TensorSpec> model_outputs_;
  std::vector<RuntimeInputSpec> runtime_input_specs_;
  std::unordered_map<std::string, StateBuffer> states_;
  std::vector<std::vector<float>> constant_inputs_;
  std::vector<std::vector<float>> output_buffers_;
  std::vector<std::vector<float>> self_check_inputs_;
  std::vector<const float*> backend_inputs_;
  std::vector<float*> backend_outputs_;
  size_t action_output_index_ = 0;
  size_t action_dim_ = 0;
  bool loaded_ = false;
};

}  // namespace legged_rl_deploy
