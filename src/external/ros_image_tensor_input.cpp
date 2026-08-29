#include "legged_rl_deploy/external/ros_image_tensor_input.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace legged_rl_deploy {

namespace {

std::pair<double, double> p95AndMax(std::vector<double> values) {
  if (values.empty()) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan};
  }
  std::sort(values.begin(), values.end());
  const size_t p95_index = static_cast<size_t>(
      std::ceil(0.95 * static_cast<double>(values.size()))) - 1;
  return {values[p95_index], values.back()};
}

}  // namespace

RosImageTensorInput::RosImageTensorInput(
    rclcpp::Node& node, const std::string& source_name,
    const YAML::Node& config, const std::vector<int64_t>& tensor_shape)
    : source_name_(source_name), logger_(node.get_logger()),
      clock_(node.get_clock()) {
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
  statistics_timer_ = node.create_wall_timer(
      std::chrono::seconds(1), [this]() { logStatistics(); });
}

void RosImageTensorInput::callback(const sensor_msgs::msg::Image& message) {
  const auto arrival_time = std::chrono::steady_clock::now();
  const int64_t arrival_clock_ns = clock_->now().nanoseconds();
  const int64_t header_stamp_ns =
      static_cast<int64_t>(message.header.stamp.sec) * 1000000000LL +
      static_cast<int64_t>(message.header.stamp.nanosec);
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

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!error.empty()) {
      error_ = std::move(error);
      received_ = false;
      return;
    }

    std::memcpy(latest_.data(), message.data.data(),
                latest_.size() * sizeof(float));
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
    latest_header_stamp_ns_ = header_stamp_ns;
    error_.clear();
    received_ = true;
  }

  std::lock_guard<std::mutex> statistics_lock(statistics_mutex_);
  arrival_times_.push_back(arrival_time);
  if (header_stamp_ns > 0) {
    callback_age_ms_.push_back(
        static_cast<double>(arrival_clock_ns - header_stamp_ns) * 1.0e-6);
  }
}

void RosImageTensorInput::read(std::vector<float>& destination) const {
  double policy_image_age_ms = std::numeric_limits<double>::quiet_NaN();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!received_) {
      throw std::runtime_error(source_name_ +
                               (error_.empty() ? " has not received an image"
                                               : ": " + error_));
    }
    if (std::chrono::steady_clock::now() - received_at_ > timeout_) {
      throw std::runtime_error(source_name_ + " image is stale");
    }
    if (latest_header_stamp_ns_ > 0) {
      policy_image_age_ms = static_cast<double>(
          clock_->now().nanoseconds() - latest_header_stamp_ns_) * 1.0e-6;
    }
    destination = latest_;
  }

  if (std::isfinite(policy_image_age_ms)) {
    std::lock_guard<std::mutex> statistics_lock(statistics_mutex_);
    policy_image_age_ms_.push_back(policy_image_age_ms);
  }
}

void RosImageTensorInput::logStatistics() {
  std::vector<std::chrono::steady_clock::time_point> arrival_times;
  std::vector<double> callback_age_ms;
  std::vector<double> policy_image_age_ms;
  {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    arrival_times.swap(arrival_times_);
    callback_age_ms.swap(callback_age_ms_);
    policy_image_age_ms.swap(policy_image_age_ms_);
  }

  double hz = 0.0;
  if (arrival_times.size() >= 2) {
    const double span_sec = std::chrono::duration<double>(
        arrival_times.back() - arrival_times.front()).count();
    if (span_sec > 0.0) {
      hz = static_cast<double>(arrival_times.size() - 1) / span_sec;
    }
  }
  const auto callback_age = p95AndMax(std::move(callback_age_ms));
  const auto policy_age = p95AndMax(std::move(policy_image_age_ms));
  RCLCPP_INFO(
      logger_,
      "[depth_input] hz=%.1f callback_age_ms(p95/max)=%.2f/%.2f "
      "policy_image_age_ms(p95/max)=%.2f/%.2f",
      hz, callback_age.first, callback_age.second, policy_age.first,
      policy_age.second);
}

}  // namespace legged_rl_deploy
