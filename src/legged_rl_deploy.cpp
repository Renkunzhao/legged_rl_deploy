#include "legged_rl_deploy/legged_rl_deploy.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <legged_base/Utils.h>

namespace legged_rl_deploy {

// ===========================================================================
// Construction
// ===========================================================================
LeggedRLDeploy::LeggedRLDeploy(std::string configFile) {
  configNode_ = YAML::LoadFile(configFile);
  std::cout << "[LeggedRLDeploy] Load config from " << configFile << std::endl;
}

// ===========================================================================
// switchToPolicy
// ===========================================================================
void LeggedRLDeploy::switchToPolicy(const std::string& name) {
  auto it = slots_.find(name);
  if (it == slots_.end()) {
    std::cerr << "[LeggedRLDeploy] Policy '" << name
              << "' not found, ignoring." << std::endl;
    return;
  }
  std::cout << "[LeggedRLDeploy] Switch: " << active_name_ << " -> " << name
            << std::endl;
  active_name_ = name;
  active_slot_ = it->second.get();
  active_slot_->reset(real_state_);
}

// ===========================================================================
// initHighController
// ===========================================================================
void LeggedRLDeploy::initHighController() {
  if (configNode_["ll_dt"]) {
    ll_dt_ = configNode_["ll_dt"].as<double>();
  }
  if (configNode_["clip_final_tau"]) {
    clip_final_tau_ = configNode_["clip_final_tau"].as<bool>();
  }
  if (configNode_["tau_max"]) {
    robot_model_.setTauMaxOrder(legged_base::yamlToEigenVec(configNode_["tau_max"]));
  }

  // ------------------------------------------------------------------
  // Detect mode: fsm (multi-policy) vs. policy (single, backward compat)
  // ------------------------------------------------------------------
  if (configNode_["fsm"]) {
    // ======== multi-policy mode ========
    single_mode_ = false;
    const auto& fsmNode = configNode_["fsm"];

    // --- Build a YAML node for GamepadFSM ---
    // GamepadFSM expects: { default: ..., states: { name: { transitions: ... }, ... } }
    YAML::Node gfsmYaml;
    gfsmYaml["default"] = fsmNode["default"].as<std::string>();

    const auto& policies = fsmNode["policies"];
    if (!policies || !policies.IsMap()) {
      throw std::runtime_error("[LeggedRLDeploy] fsm.policies must be a map");
    }

    YAML::Node statesYaml;

    for (auto it = policies.begin(); it != policies.end(); ++it) {
      const std::string pname = it->first.as<std::string>();
      const auto& entry = it->second;

      // --- load PolicySlot ---
      YAML::Node policyNode;
      if (entry["config"]) {
        const std::string sub_path =
            legged_base::getEnv("WORKSPACE") + "/" +
            entry["config"].as<std::string>();
        YAML::Node sub = YAML::LoadFile(sub_path);
        policyNode = sub["policy"];
        std::cout << "[LeggedRLDeploy] Load sub-config for '" << pname
                  << "' from " << sub_path << std::endl;
      } else if (entry["policy"]) {
        policyNode = entry["policy"];
      } else {
        throw std::runtime_error(
            "[LeggedRLDeploy] fsm.policies." + pname +
            " must have 'config' (path) or inline 'policy' node");
      }

      auto slot =
          std::make_unique<PolicySlot>(pname, policyNode, robot_model_);
      slot->init();
      slots_.emplace(pname, std::move(slot));

      // --- forward transitions to GamepadFSM YAML ---
      YAML::Node stateEntry;
      if (entry["transitions"] && entry["transitions"].IsMap()) {
        stateEntry["transitions"] = entry["transitions"];
      }
      statesYaml[pname] = stateEntry;
    }

    gfsmYaml["states"] = statesYaml;

    // --- init GamepadFSM ---
    policy_fsm_.loadFromYAML(gfsmYaml);
    policy_fsm_.setOnTransition(
        [this](const std::string& /*from*/, const std::string& to) {
          switchToPolicy(to);
        });

    active_name_ = policy_fsm_.activeState();
    active_slot_ = slots_[active_name_].get();

    std::cout << "[LeggedRLDeploy] Multi-policy mode: " << slots_.size()
              << " policies loaded. default='" << active_name_ << "'"
              << std::endl;

  } else if (configNode_["policy"]) {
    // ======== single-policy mode (backward compatible) ========
    single_mode_ = true;
    const std::string pname = "default";
    auto slot = std::make_unique<PolicySlot>(
        pname, configNode_["policy"], robot_model_);
    slot->init();
    active_name_ = pname;
    active_slot_ = slot.get();
    slots_.emplace(pname, std::move(slot));

    std::cout << "[LeggedRLDeploy] Single-policy mode (backward compat)."
              << std::endl;

  } else {
    throw std::runtime_error(
        "[LeggedRLDeploy] Config must have 'fsm' or 'policy' section");
  }

  std::cout << "[LeggedRLDeploy] init done." << std::endl;
}

// ===========================================================================
// resetHighController
// ===========================================================================
void LeggedRLDeploy::resetHighController() {
  if (!single_mode_) {
    switchToPolicy(policy_fsm_.defaultState());
    policy_fsm_.setState(policy_fsm_.defaultState());
  } else if (active_slot_) {
    active_slot_->reset(real_state_);
  }
}

// ===========================================================================
// updateHighController
// ===========================================================================
void LeggedRLDeploy::updateHighController() {
  // 1) Check FSM transitions (multi-policy mode only)
  if (!single_mode_) {
    policy_fsm_.update(gamepad_);  // callback fires switchToPolicy if needed
  }

  // 2) Run active policy
  active_slot_->update(real_state_, gamepad_, loop_cnt_, ll_dt_);

  // 3) Write joint commands from active slot's output
  const auto& out = active_slot_->outputBuf();
  const auto& jmap = active_slot_->jointIdsMap();
  const auto& kp = active_slot_->stiffness();
  const auto& kd = active_slot_->damping();
  const size_t n = active_slot_->outputDim();

  for (size_t i = 0; i < n; ++i) {
    const size_t j = jmap[i];

    if (clip_final_tau_) {
      jnt_cmd_.tau[j] = kp[i] * (out[i] - real_state_.joint_pos()[j]) +
                        kd[i] * (0.0f - real_state_.joint_vel()[j]);
      jnt_cmd_.q[j] = 0.0f;
      jnt_cmd_.dq[j] = 0.0f;
      jnt_cmd_.kp[j] = 0.0f;
      jnt_cmd_.kd[j] = 0.0f;
    } else {
      jnt_cmd_.q[j] = out[i];
      jnt_cmd_.kp[j] = kp[i];
      jnt_cmd_.kd[j] = kd[i];
      jnt_cmd_.dq[j] = 0.0f;
      jnt_cmd_.tau[j] = 0.0f;
    }
  }
}

} // namespace legged_rl_deploy
