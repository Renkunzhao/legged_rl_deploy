#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <yaml-cpp/yaml.h>

#include "legged_rl_deploy/external/tensor_input.h"

namespace legged_rl_deploy {

class RosImageTensorInput final : public TensorInput {
public:
  RosImageTensorInput(rclcpp::Node& node, const std::string& source_name,
                      const YAML::Node& config,
                      const std::vector<int64_t>& tensor_shape);

  void read(std::vector<float>& destination) const override;

private:
  void callback(const sensor_msgs::msg::Image& message);
  void logStatistics();

  std::string source_name_;
  std::string encoding_;
  rclcpp::Logger logger_;
  rclcpp::Clock::SharedPtr clock_;
  size_t height_ = 0;
  size_t width_ = 0;
  std::chrono::duration<double> timeout_;
  bool check_range_ = false;
  float min_value_ = 0.0f;
  float max_value_ = 0.0f;

  mutable std::mutex mutex_;
  std::vector<float> latest_;
  std::chrono::steady_clock::time_point received_at_{};
  int64_t latest_header_stamp_ns_ = 0;
  bool received_ = false;
  std::string error_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr statistics_timer_;

  mutable std::mutex statistics_mutex_;
  std::vector<std::chrono::steady_clock::time_point> arrival_times_;
  std::vector<double> callback_age_ms_;
  mutable std::vector<double> policy_image_age_ms_;
};

}  // namespace legged_rl_deploy
