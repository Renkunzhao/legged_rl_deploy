#include "legged_rl_deploy/policy/i_policy_runner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace legged_rl_deploy {
namespace {

bool startsWith(const std::string& value, const char* prefix) {
  return value.rfind(prefix, 0) == 0;
}

std::string stateName(const std::string& binding) {
  return binding.substr(std::string("state.").size());
}

std::vector<int64_t> loadShape(const YAML::Node& node,
                               const std::string& context) {
  if (!node || !node.IsSequence() || node.size() == 0) {
    throw std::runtime_error(context + ".shape must be a non-empty sequence");
  }
  return node.as<std::vector<int64_t>>();
}

}  // namespace

size_t IPolicyRunner::tensorSize(const std::vector<int64_t>& shape,
                                 const std::string& context) {
  size_t size = 1;
  for (const int64_t dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error(context + " must have positive static dimensions");
    }
    if (size > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      throw std::runtime_error(context + " is too large");
    }
    size *= static_cast<size_t>(dim);
  }
  return size;
}

void IPolicyRunner::parseContract(const YAML::Node& policy_node) {
  model_inputs_.clear();
  model_outputs_.clear();
  runtime_input_specs_.clear();
  states_.clear();
  constant_inputs_.clear();
  action_output_index_ = 0;

  const YAML::Node model = policy_node["model"];
  if (!model) {
    const size_t input_dim = policy_node["input_dim"].as<size_t>();
    const size_t output_dim = policy_node["output_dim"].as<size_t>();
    if (input_dim == 0 || output_dim == 0) {
      throw std::runtime_error("policy input_dim and output_dim must be > 0");
    }
    model_inputs_.push_back(
        {"", {1, static_cast<int64_t>(input_dim)}, input_dim, "observations"});
    constant_inputs_.emplace_back();
    model_outputs_.push_back(
        {"", {1, static_cast<int64_t>(output_dim)}, output_dim, "actions"});
  } else {
    if (policy_node["input_dim"] || policy_node["output_dim"]) {
      throw std::runtime_error(
          "policy.input_dim/output_dim must not be used with policy.model");
    }
    if (!model.IsMap()) {
      throw std::runtime_error("policy.model must be a map");
    }

    const YAML::Node states = model["states"];
    if (states) {
      if (!states.IsMap()) {
        throw std::runtime_error("policy.model.states must be a map");
      }
      for (auto it = states.begin(); it != states.end(); ++it) {
        const std::string name = it->first.as<std::string>();
        if (name.empty()) throw std::runtime_error("state name must not be empty");
        const YAML::Node spec = it->second;
        if (!spec.IsMap()) {
          throw std::runtime_error("policy.model.states." + name + " must be a map");
        }
        const std::string initialization =
            spec["initialization"].as<std::string>("zeros");
        if (initialization != "zeros") {
          throw std::runtime_error("policy.model.states." + name +
                                   ".initialization must be zeros");
        }
        StateBuffer state;
        state.shape = loadShape(spec["shape"], "policy.model.states." + name);
        const size_t size = tensorSize(state.shape, "policy.model.states." + name);
        state.current.assign(size, 0.0f);
        state.next.assign(size, 0.0f);
        state.max_norm = spec["max_norm"].as<float>(0.0f);
        if (state.max_norm < 0.0f || !std::isfinite(state.max_norm)) {
          throw std::runtime_error("policy.model.states." + name +
                                   ".max_norm must be finite and >= 0");
        }
        states_.emplace(name, std::move(state));
      }
    }

    const YAML::Node inputs = model["inputs"];
    const YAML::Node outputs = model["outputs"];
    if (!inputs || !inputs.IsSequence() || inputs.size() == 0) {
      throw std::runtime_error("policy.model.inputs must be a non-empty sequence");
    }
    if (!outputs || !outputs.IsSequence() || outputs.size() == 0) {
      throw std::runtime_error("policy.model.outputs must be a non-empty sequence");
    }

    std::unordered_set<std::string> names;
    for (size_t i = 0; i < inputs.size(); ++i) {
      const YAML::Node spec = inputs[i];
      if (!spec.IsMap() || !spec["name"] || !spec["source"]) {
        throw std::runtime_error("each policy.model.inputs entry requires name and source");
      }
      TensorSpec tensor;
      tensor.name = spec["name"].as<std::string>();
      tensor.binding = spec["source"].as<std::string>();
      if (tensor.name.empty() || !names.emplace(tensor.name).second) {
        throw std::runtime_error("policy.model.inputs names must be non-empty and unique");
      }
      if (startsWith(tensor.binding, "state.")) {
        const auto state = states_.find(stateName(tensor.binding));
        if (state == states_.end()) {
          throw std::runtime_error("unknown input source " + tensor.binding);
        }
        if (spec["shape"]) {
          throw std::runtime_error("state input shape is declared only in model.states");
        }
        tensor.shape = state->second.shape;
        constant_inputs_.emplace_back();
      } else if (tensor.binding == "constant") {
        tensor.shape = loadShape(spec["shape"], "policy.model.inputs[" +
                                                    std::to_string(i) + "]");
        const size_t size = tensorSize(tensor.shape, "model input " + tensor.name);
        if (!spec["value"]) {
          throw std::runtime_error("constant input " + tensor.name +
                                   " requires value");
        }
        std::vector<float> values;
        if (spec["value"].IsScalar()) {
          values.assign(size, spec["value"].as<float>());
        } else if (spec["value"].IsSequence()) {
          values = spec["value"].as<std::vector<float>>();
          if (values.size() != size) {
            throw std::runtime_error("constant input " + tensor.name +
                                     " value size does not match shape");
          }
        } else {
          throw std::runtime_error("constant input " + tensor.name +
                                   " value must be a scalar or sequence");
        }
        if (!std::all_of(values.begin(), values.end(),
                         [](float value) { return std::isfinite(value); })) {
          throw std::runtime_error("constant input " + tensor.name +
                                   " contains NaN/Inf");
        }
        constant_inputs_.push_back(std::move(values));
      } else {
        if (tensor.binding != "observations" &&
            !startsWith(tensor.binding, "external.")) {
          throw std::runtime_error("unsupported input source " + tensor.binding);
        }
        tensor.shape = loadShape(spec["shape"], "policy.model.inputs[" +
                                                    std::to_string(i) + "]");
        constant_inputs_.emplace_back();
      }
      tensor.size = tensorSize(tensor.shape, "model input " + tensor.name);
      model_inputs_.push_back(std::move(tensor));
    }

    names.clear();
    size_t action_count = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
      const YAML::Node spec = outputs[i];
      if (!spec.IsMap() || !spec["name"] || !spec["target"]) {
        throw std::runtime_error("each policy.model.outputs entry requires name and target");
      }
      TensorSpec tensor;
      tensor.name = spec["name"].as<std::string>();
      tensor.binding = spec["target"].as<std::string>();
      if (tensor.name.empty() || !names.emplace(tensor.name).second) {
        throw std::runtime_error("policy.model.outputs names must be non-empty and unique");
      }
      if (startsWith(tensor.binding, "state.")) {
        const auto state = states_.find(stateName(tensor.binding));
        if (state == states_.end()) {
          throw std::runtime_error("unknown output target " + tensor.binding);
        }
        if (spec["shape"]) {
          throw std::runtime_error("state output shape is declared only in model.states");
        }
        tensor.shape = state->second.shape;
      } else if (tensor.binding == "actions") {
        tensor.shape = loadShape(spec["shape"], "policy.model.outputs[" +
                                                    std::to_string(i) + "]");
        action_output_index_ = i;
        ++action_count;
      } else if (tensor.binding == "discard") {
        tensor.shape = loadShape(spec["shape"], "policy.model.outputs[" +
                                                    std::to_string(i) + "]");
      } else {
        throw std::runtime_error("unsupported output target " + tensor.binding);
      }
      tensor.size = tensorSize(tensor.shape, "model output " + tensor.name);
      model_outputs_.push_back(std::move(tensor));
    }
    if (action_count != 1) {
      throw std::runtime_error("policy.model.outputs must contain exactly one actions target");
    }
  }

  if (!model) action_output_index_ = 0;
  action_dim_ = model_outputs_[action_output_index_].size;

  std::unordered_set<std::string> runtime_sources;
  for (const auto& input : model_inputs_) {
    if (startsWith(input.binding, "state.") || input.binding == "constant") continue;
    if (!runtime_sources.emplace(input.binding).second) {
      throw std::runtime_error("runtime input source used more than once: " +
                               input.binding);
    }
    runtime_input_specs_.push_back({input.binding, input.shape, input.size});
  }

  std::unordered_map<std::string, size_t> state_inputs;
  std::unordered_map<std::string, size_t> state_outputs;
  for (const auto& input : model_inputs_) {
    if (startsWith(input.binding, "state.")) ++state_inputs[input.binding];
  }
  for (const auto& output : model_outputs_) {
    if (startsWith(output.binding, "state.")) ++state_outputs[output.binding];
  }
  for (const auto& entry : states_) {
    const std::string binding = "state." + entry.first;
    if (state_inputs[binding] != 1 || state_outputs[binding] != 1) {
      throw std::runtime_error(binding +
                               " must be bound to exactly one input and one output");
    }
  }
}

void IPolicyRunner::load(const std::string& model_path,
                         const YAML::Node& policy_node) {
  loaded_ = false;
  parseContract(policy_node);
  loadBackend(model_path);

  output_buffers_.clear();
  backend_outputs_.clear();
  output_buffers_.reserve(model_outputs_.size());
  backend_outputs_.reserve(model_outputs_.size());
  for (const auto& output : model_outputs_) {
    output_buffers_.emplace_back(output.size, 0.0f);
    backend_outputs_.push_back(output_buffers_.back().data());
  }
  self_check_inputs_.clear();
  self_check_inputs_.reserve(runtime_input_specs_.size());
  for (const auto& input : runtime_input_specs_) {
    self_check_inputs_.emplace_back(input.size, 0.0f);
  }
  std::vector<RuntimeTensor> self_check;
  self_check.reserve(runtime_input_specs_.size());
  for (size_t i = 0; i < runtime_input_specs_.size(); ++i) {
    const auto& input = runtime_input_specs_[i];
    self_check.push_back({input.source, self_check_inputs_[i].data(), input.size});
  }
  loaded_ = true;
  std::vector<float> actions(action_dim_, 0.0f);
  run(self_check, actions.data());
  reset();
}

void IPolicyRunner::run(const std::vector<RuntimeTensor>& runtime_inputs,
                        float* actions) {
  backend_inputs_.clear();
  for (size_t i = 0; i < model_inputs_.size(); ++i) {
    const auto& input = model_inputs_[i];
    if (startsWith(input.binding, "state.")) {
      backend_inputs_.push_back(states_.at(stateName(input.binding)).current.data());
      continue;
    }
    if (input.binding == "constant") {
      backend_inputs_.push_back(constant_inputs_[i].data());
      continue;
    }
    const auto runtime = std::find_if(
        runtime_inputs.begin(), runtime_inputs.end(), [&](const RuntimeTensor& value) {
          return value.source == input.binding;
        });
    if (runtime == runtime_inputs.end()) {
      throw std::runtime_error("missing runtime input source " + input.binding);
    }
    if (!runtime->data || runtime->size != input.size) {
      throw std::runtime_error("runtime input size mismatch for " + input.binding);
    }
    if (!std::all_of(runtime->data, runtime->data + runtime->size,
                     [](float value) { return std::isfinite(value); })) {
      throw std::runtime_error("runtime input " + input.binding +
                               " contains NaN/Inf");
    }
    backend_inputs_.push_back(runtime->data);
  }
  if (runtime_inputs.size() != runtime_input_specs_.size()) {
    throw std::runtime_error("unexpected runtime input source");
  }

  runBackend(backend_inputs_, backend_outputs_);

  for (size_t i = 0; i < model_outputs_.size(); ++i) {
    const auto& output = model_outputs_[i];
    const auto& values = output_buffers_[i];
    if (!std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); })) {
      throw std::runtime_error("model output " + output.name + " contains NaN/Inf");
    }
    if (startsWith(output.binding, "state.")) {
      StateBuffer& state = states_.at(stateName(output.binding));
      if (state.max_norm > 0.0f) {
        double squared_norm = 0.0;
        for (const float value : values) squared_norm += value * value;
        if (std::sqrt(squared_norm) > state.max_norm) {
          throw std::runtime_error(output.binding + " exceeds max_norm");
        }
      }
      std::copy(values.begin(), values.end(), state.next.begin());
    }
  }

  const auto& action_values = output_buffers_[action_output_index_];
  std::copy(action_values.begin(), action_values.end(), actions);
  for (auto& entry : states_) entry.second.current.swap(entry.second.next);
}

void IPolicyRunner::infer(const std::vector<RuntimeTensor>& runtime_inputs,
                          float* actions) {
  if (!loaded_) throw std::runtime_error("policy runner is not loaded");
  if (!actions) throw std::runtime_error("actions buffer is null");
  run(runtime_inputs, actions);
}

void IPolicyRunner::reset() {
  if (!loaded_) return;
  for (auto& entry : states_) {
    std::fill(entry.second.current.begin(), entry.second.current.end(), 0.0f);
    std::fill(entry.second.next.begin(), entry.second.next.end(), 0.0f);
  }
  resetBackend();
}

size_t IPolicyRunner::observationDim() const {
  const auto it = std::find_if(runtime_input_specs_.begin(), runtime_input_specs_.end(),
                               [](const RuntimeInputSpec& input) {
                                 return input.source == "observations";
                               });
  if (it == runtime_input_specs_.end()) return 0;
  return it->size;
}

}  // namespace legged_rl_deploy
