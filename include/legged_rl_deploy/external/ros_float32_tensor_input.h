#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <yaml-cpp/yaml.h>

#include "legged_rl_deploy/external/tensor_input.h"

namespace legged_rl_deploy {

class RosFloat32TensorInput final : public TensorInput {
public:
  RosFloat32TensorInput(rclcpp::Node& node, const std::string& source_name,
                        const YAML::Node& config,
                        const std::vector<int64_t>& tensor_shape);

  void read(std::vector<float>& destination) const override;

private:
  void callback(const std_msgs::msg::Float32MultiArray& message);

  std::string source_name_;
  size_t size_ = 0;
  std::chrono::duration<double> timeout_;

  mutable std::mutex mutex_;
  std::vector<float> latest_;
  std::chrono::steady_clock::time_point received_at_{};
  bool received_ = false;
  std::string error_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr subscription_;
};

}  // namespace legged_rl_deploy
