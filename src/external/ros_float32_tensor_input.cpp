#include "legged_rl_deploy/external/ros_float32_tensor_input.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace legged_rl_deploy {

RosFloat32TensorInput::RosFloat32TensorInput(
    rclcpp::Node& node, const std::string& source_name,
    const YAML::Node& config, const std::vector<int64_t>& tensor_shape)
    : source_name_(source_name) {
  if (!config || !config.IsMap()) {
    throw std::runtime_error(source_name_ + " external input config must be a map");
  }
  if (config["type"].as<std::string>() != "ros_float32_multi_array") {
    throw std::runtime_error(source_name_ +
                             " type must be ros_float32_multi_array");
  }
  if (tensor_shape.size() != 2 || tensor_shape[0] != 1 ||
      tensor_shape[1] <= 0) {
    throw std::runtime_error(source_name_ +
                             " ros_float32_multi_array shape must be [1, N]");
  }
  size_ = static_cast<size_t>(tensor_shape[1]);

  const double timeout_sec = config["timeout_sec"].as<double>();
  if (!std::isfinite(timeout_sec) || timeout_sec <= 0.0) {
    throw std::runtime_error(source_name_ + " timeout_sec must be finite and > 0");
  }
  timeout_ = std::chrono::duration<double>(timeout_sec);

  const std::string topic = config["topic"].as<std::string>();
  if (topic.empty()) {
    throw std::runtime_error(source_name_ + " topic must not be empty");
  }
  latest_.assign(size_, 0.0f);
  auto qos = rclcpp::SensorDataQoS();
  qos.keep_last(1);
  subscription_ = node.create_subscription<std_msgs::msg::Float32MultiArray>(
      topic, qos,
      [this](const std_msgs::msg::Float32MultiArray::SharedPtr message) {
        callback(*message);
      });
}

void RosFloat32TensorInput::callback(
    const std_msgs::msg::Float32MultiArray& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (message.data.size() != size_) {
    error_ = "array length is " + std::to_string(message.data.size()) +
             ", expected " + std::to_string(size_);
    received_ = false;
    return;
  }
  if (!std::all_of(message.data.begin(), message.data.end(),
                   [](float value) { return std::isfinite(value); })) {
    error_ = "array contains NaN/Inf";
    received_ = false;
    return;
  }
  latest_ = message.data;
  received_at_ = std::chrono::steady_clock::now();
  error_.clear();
  received_ = true;
}

void RosFloat32TensorInput::read(std::vector<float>& destination) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!received_) {
    throw std::runtime_error(
        source_name_ +
        (error_.empty() ? " has not received an array" : ": " + error_));
  }
  if (std::chrono::steady_clock::now() - received_at_ > timeout_) {
    throw std::runtime_error(source_name_ + " array is stale");
  }
  destination = latest_;
}

}  // namespace legged_rl_deploy
