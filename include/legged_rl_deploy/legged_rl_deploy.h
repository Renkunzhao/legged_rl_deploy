#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include <yaml-cpp/yaml.h>

#include "legged_rl_deploy/msg/deploy_status.hpp"
#include "legged_rl_deploy/policy_slot.h"
#include <unitree_lowlevel/gamepad_fsm.hpp>
#include <unitree_lowlevel/lowlevel_controller.h>

namespace legged_rl_deploy {

class LeggedRLDeploy : public LowLevelController {
public:
  LeggedRLDeploy(std::string configFile);
  ~LeggedRLDeploy();

private:
  void initHighController() override;
  void resetHighController() override;
  void updateHighController() override;
  void log() override;

  void switchToPolicy(const std::string& name);
  void publishStatus(bool force);

  // -------- config --------
  YAML::Node configNode_;
  bool clip_final_tau_ = false;

  // -------- multi-policy slots --------
  std::unordered_map<std::string, std::unique_ptr<PolicySlot>> slots_;
  PolicySlot* active_slot_ = nullptr;
  std::string active_name_;

  // -------- FSM (from unitree_lowlevel, reusable) --------
  unitree::common::GamepadFSM policy_fsm_;

  // -------- backward compat: single-policy mode --------
  bool single_mode_ = false;

  // -------- read-only evaluation status --------
  bool evaluation_mode_ = false;
  uint64_t policy_reset_sequence_ = 0;
  bool policy_output_valid_ = false;
  std::string last_fault_;
  rclcpp::Publisher<msg::DeployStatus>::SharedPtr status_publisher_;
  std::chrono::steady_clock::time_point last_status_publish_{};
  RobotState previous_logged_state_ = RobotState::IDLE;
};

} // namespace legged_rl_deploy
