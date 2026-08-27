#include "legged_rl_deploy/external/ros_image_tensor_input.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace legged_rl_deploy {

RosImageTensorInput::RosImageTensorInput(
    rclcpp::Node& node, const std::string& source_name,
    const YAML::Node& config, const std::vector<int64_t>& tensor_shape)
    : source_name_(source_name) {
  if (!config || !config.IsMap()) {
    throw std::runtime_error(source_name_ + " external input config must be a map");
  }
  if (config["type"].as<std::string>() != "ros_image") {
    throw std::runtime_error(source_name_ + " type must be ros_image");
  }
  if (tensor_shape.size() != 4 || tensor_shape[0] != 1 ||
      tensor_shape[1] != 1) {
    throw std::runtime_error(source_name_ + " ros_image shape must be [1, 1, H, W]");
  }
  height_ = static_cast<size_t>(tensor_shape[2]);
  width_ = static_cast<size_t>(tensor_shape[3]);
  encoding_ = config["encoding"].as<std::string>();
  if (encoding_ != "32FC1") {
    throw std::runtime_error(source_name_ + " encoding must be 32FC1");
  }
  const double timeout_sec = config["timeout_sec"].as<double>();
  if (!std::isfinite(timeout_sec) || timeout_sec <= 0.0) {
    throw std::runtime_error(source_name_ + " timeout_sec must be finite and > 0");
  }
  timeout_ = std::chrono::duration<double>(timeout_sec);

  if (config["value_range"]) {
    const auto range = config["value_range"].as<std::vector<float>>();
    if (range.size() != 2 || !std::isfinite(range[0]) ||
        !std::isfinite(range[1]) || range[0] > range[1]) {
      throw std::runtime_error(source_name_ +
                               " value_range must be [finite_min, finite_max]");
    }
    check_range_ = true;
    min_value_ = range[0];
    max_value_ = range[1];
  }

  latest_.assign(height_ * width_, 0.0f);
  const std::string topic = config["topic"].as<std::string>();
  if (topic.empty()) throw std::runtime_error(source_name_ + " topic must not be empty");
  auto image_qos = rclcpp::SensorDataQoS();
  image_qos.keep_last(1);
  subscription_ = node.create_subscription<sensor_msgs::msg::Image>(
      topic, image_qos,
      [this](const sensor_msgs::msg::Image::SharedPtr message) {
        callback(*message);
      });
}

void RosImageTensorInput::callback(const sensor_msgs::msg::Image& message) {
  std::string error;
  if (message.encoding != encoding_) {
    error = "encoding is " + message.encoding + ", expected " + encoding_;
  } else if (message.is_bigendian) {
    error = "big-endian images are unsupported";
  } else if (message.height != height_ || message.width != width_) {
    error = "image dimensions do not match model input";
  } else if (message.step != width_ * sizeof(float) ||
             message.data.size() != height_ * message.step) {
    error = "image buffer is not tightly packed 32FC1";
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!error.empty()) {
    error_ = std::move(error);
    received_ = false;
    return;
  }

  std::memcpy(latest_.data(), message.data.data(), latest_.size() * sizeof(float));
  for (const float value : latest_) {
    if (!std::isfinite(value)) {
      error_ = "image contains NaN/Inf";
      received_ = false;
      return;
    }
    if (check_range_ && (value < min_value_ || value > max_value_)) {
      error_ = "image value is outside configured value_range";
      received_ = false;
      return;
    }
  }
  received_at_ = std::chrono::steady_clock::now();
  error_.clear();
  received_ = true;
}

void RosImageTensorInput::read(std::vector<float>& destination) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!received_) {
    throw std::runtime_error(source_name_ +
                             (error_.empty() ? " has not received an image"
                                             : ": " + error_));
  }
  if (std::chrono::steady_clock::now() - received_at_ > timeout_) {
    throw std::runtime_error(source_name_ + " image is stale");
  }
  destination = latest_;
}

}  // namespace legged_rl_deploy
