#include "legged_rl_deploy/legged_rl_deploy.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

#include <legged_base/Utils.h>

namespace legged_rl_deploy {

namespace {

void applyPolicyFileOverrides(const std::string& pname, const YAML::Node& entry,
                              YAML::Node& policy_node) {
  const bool has_model_file = entry["model_file"] && !entry["model_file"].IsNull();
  const bool has_motion_file = entry["motion_file"] && !entry["motion_file"].IsNull();

  if (!has_model_file && !has_motion_file) {
    return;
  }

  std::string model_file;
  if (has_model_file) {
    model_file = entry["model_file"].as<std::string>();
    policy_node["model_path"] = model_file;
    std::cout << "[LeggedRLDeploy] Override '" << pname
              << "': policy.model_path <- " << model_file << std::endl;
  }

  std::string effective_motion_file;
  bool motion_is_explicit = false;
  if (has_motion_file) {
    effective_motion_file = entry["motion_file"].as<std::string>();
    motion_is_explicit = true;
  } else if (has_model_file) {
    effective_motion_file = model_file;
  }

  if (effective_motion_file.empty()) {
    return;
  }

  const YAML::Node policy_const(policy_node);
  const YAML::Node observations = policy_const["observations"];
  const YAML::Node terms = observations ? observations["terms"] : YAML::Node();
  const YAML::Node mimic_const = terms ? terms["mimic"] : YAML::Node();

  if (!mimic_const || !mimic_const.IsMap()) {
    if (motion_is_explicit) {
      throw std::runtime_error(
          "[LeggedRLDeploy] fsm.policies." + pname +
          ".motion_file is set, but policy has no observations.terms.mimic");
    }
    std::cout << "[LeggedRLDeploy] Override '" << pname
              << "': auto motion sync skipped (no mimic term)." << std::endl;
    return;
  }

  const YAML::Node params_const = mimic_const["params"];
  if (params_const && !params_const.IsMap()) {
    throw std::runtime_error("[LeggedRLDeploy] fsm.policies." + pname +
                             ".observations.terms.mimic.params must be a map");
  }

  const std::string source =
      params_const ? params_const["source"].as<std::string>("local") : "local";
  if (source != "local") {
    if (motion_is_explicit) {
      throw std::runtime_error("[LeggedRLDeploy] fsm.policies." + pname +
                               ".motion_file requires mimic.params.source=local");
    }
    std::cout << "[LeggedRLDeploy] Override '" << pname
              << "': auto motion sync skipped (mimic source='" << source
              << "')." << std::endl;
    return;
  }

  YAML::Node mimic = policy_node["observations"]["terms"]["mimic"];
  YAML::Node params = mimic["params"];
  if (!params || params.IsNull()) {
    mimic["params"] = YAML::Node(YAML::NodeType::Map);
    params = mimic["params"];
  }
  YAML::Node local = params["local"];
  if (!local || local.IsNull()) {
    params["local"] = YAML::Node(YAML::NodeType::Map);
    local = params["local"];
  }
  if (!local.IsMap()) {
    throw std::runtime_error("[LeggedRLDeploy] fsm.policies." + pname +
                             ".observations.terms.mimic.params.local must be a map");
  }

  local["file"] = effective_motion_file;
  std::cout << "[LeggedRLDeploy] Override '" << pname
            << "': mimic.params.local.file <- " << effective_motion_file
            << (motion_is_explicit ? " (explicit motion_file)"
                                   : " (auto-sync from model_file)")
            << std::endl;
}

} // namespace

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
      applyPolicyFileOverrides(pname, entry, policyNode);

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
