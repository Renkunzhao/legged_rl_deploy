#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "legged_rl_deploy/policy_slot.h"
#include <unitree_lowlevel/gamepad_fsm.hpp>
#include <unitree_lowlevel/lowlevel_controller.h>

namespace legged_rl_deploy {

class LeggedRLDeploy : public LowLevelController {
public:
  LeggedRLDeploy(std::string configFile);

private:
  void initHighController() override;
  void resetHighController() override;
  void updateHighController() override;

  void switchToPolicy(const std::string& name);

  // -------- config --------
  YAML::Node configNode_;
  bool clip_final_tau_ = true;

  // -------- multi-policy slots --------
  std::unordered_map<std::string, std::unique_ptr<PolicySlot>> slots_;
  PolicySlot* active_slot_ = nullptr;
  std::string active_name_;

  // -------- FSM (from unitree_lowlevel, reusable) --------
  unitree::common::GamepadFSM policy_fsm_;

  // -------- backward compat: single-policy mode --------
  bool single_mode_ = false;
};

} // namespace legged_rl_deploy
