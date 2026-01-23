#include "legged_rl_deploy/legged_rl_deploy.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "legged_rl_deploy/policy/policy_factory.h"

namespace legged_rl_deploy {

LeggedRLDeploy::LeggedRLDeploy(std::string configFile) {
  configNode_ = YAML::LoadFile(configFile);
  std::cout << "[LeggedRLDeploy] Load config from " << configFile << std::endl;
}

void LeggedRLDeploy::initHighController() {
  const auto& pnode = configNode_["policy"];
  const std::string backend    = pnode["backend"].as<std::string>("torch");
  const std::string model_path = pnode["model_path"].as<std::string>();

  obs_dim_ = pnode["obs_dim"].as<size_t>();
  act_dim_ = pnode["act_dim"].as<size_t>();

  policy_runner_ = makePolicyRunner(backend);
  policy_runner_->load(model_path, obs_dim_, act_dim_);
  std::cout << "[LeggedRLDeploy] Policy load successfully." << std::endl;

  obs_buf_.assign(obs_dim_, 0.0f);
  act_buf_.assign(act_dim_, 0.0f);
  last_action_.assign(act_dim_, 0.0f);

  policy_dt_ = pnode["policy_dt"].as<float>(0.02f);
  joint_ids_map_ = pnode["joint_ids_map"].as<std::vector<size_t>>();
  stiffness_ = pnode["stiffness"].as<std::vector<float>>();
  damping_ = pnode["damping"].as<std::vector<float>>();

  // -------- commands --------
  if (pnode["commands"]) {
    for (auto it = pnode["commands"].begin(); it != pnode["commands"].end(); ++it) {
      const std::string name = it->first.as<std::string>();
      auto maybe = Processor::TryLoad(it->second);
      if (maybe) commands_.emplace(name, std::move(*maybe));
    }
  }

  // -------- actions --------
  if (pnode["actions"]) {
    for (auto it = pnode["actions"].begin(); it != pnode["actions"].end(); ++it) {
      const std::string name = it->first.as<std::string>();
      auto maybe = Processor::TryLoad(it->second);
      if (maybe) actions_.emplace(name, std::move(*maybe));
    }
  }

  registryObsTerms(pnode["observations"]["terms"]);

  std::cout << "[LeggedRLDeploy] init done. obs_dim=" << obs_dim_
            << " act_dim=" << act_dim_ << std::endl;
}

void LeggedRLDeploy::resetHighController() {
  std::fill(obs_buf_.begin(), obs_buf_.end(), 0.0f);
  std::fill(act_buf_.begin(), act_buf_.end(), 0.0f);
  std::fill(last_action_.begin(), last_action_.end(), 0.0f);
}

void LeggedRLDeploy::registryObsTerms(YAML::Node node) {
  // -------- observations (from yaml, order preserved) --------
  obs_terms_.clear();
  size_t off = 0;

  const auto terms = node;
  if (!terms || !terms.IsMap()) {
    throw std::runtime_error("[LeggedRLDeploy] policy.observations.terms missing or not a map");
  }

  for (auto it = terms.begin(); it != terms.end(); ++it) {
    ObsTerm t;
    t.name = it->first.as<std::string>();
    t.dim = DimOfObsTerm(t.name);
    t.offset = off;
    off += t.dim;

    auto maybe = Processor::TryLoad(it->second);  // 没有 process_order => nullopt
    if (maybe) t.proc = std::move(*maybe);

    obs_terms_.push_back(std::move(t));
  }

  if (off != obs_dim_) {
    throw std::runtime_error("[LeggedRLDeploy] obs_dim mismatch: built=" +
                             std::to_string(off) + " cfg obs_dim=" +
                             std::to_string(obs_dim_));
  }
}

void LeggedRLDeploy::assembleObs() {
  // -------- build obs by terms order --------
  for (const auto& t : obs_terms_) {
    std::vector<float> v(t.dim, 0.0f);

    if (t.name == "base_ang_vel") {
      v[0] = lowstate_msg_.imu_state.gyroscope[0];
      v[1] = lowstate_msg_.imu_state.gyroscope[1];
      v[2] = lowstate_msg_.imu_state.gyroscope[2];
    }
    else if (t.name == "projected_gravity") {
      Eigen::Vector3d projected_gravity = real_state_.base_quat().conjugate() * Eigen::Vector3d(0.0f, 0.0f, -1.0f);
      v[0] = projected_gravity[0];
      v[1] = projected_gravity[1];
      v[2] = projected_gravity[2];
    }
    else if (t.name == "velocity_commands") {
      v = {gamepad_.ly, -gamepad_.lx, -gamepad_.rx};
      commands_["base_velocity"].process(v);
    }
    else if (t.name == "joint_pos") {
      for (size_t i = 0; i < robot_model_.nJoints(); ++i) v[i] = lowstate_msg_.motor_state[joint_ids_map_[i]].q;
    }
    else if (t.name == "joint_vel") {
      for (size_t i = 0; i < robot_model_.nJoints(); ++i) v[i] = lowstate_msg_.motor_state[joint_ids_map_[i]].dq;
    }
    else if (t.name == "last_action") {
      v = last_action_; // prev action
    }

    if (t.proc) t.proc->process(v);

    std::copy(v.begin(), v.end(), obs_buf_.begin() + t.offset);
  }
}

void LeggedRLDeploy::updateHighController() {
  std::fill(obs_buf_.begin(), obs_buf_.end(), 0.0f);

  // -------- observation assemble --------
  assembleObs();

  // -------- policy infer --------
  policy_runner_->infer(obs_buf_.data(), act_buf_.data());

  // -------- update last_action (prev) --------
  last_action_ = act_buf_;

  // -------- action process --------
  actions_["JointPositionAction"].process(act_buf_);

  // -------- send lowcmd --------
  for (size_t i = 0; i < act_dim_; ++i) {
    lowcmd_msg_.motor_cmd[joint_ids_map_[i]].q  = act_buf_[i];
    lowcmd_msg_.motor_cmd[joint_ids_map_[i]].kp = stiffness_[i];
    lowcmd_msg_.motor_cmd[joint_ids_map_[i]].kd = damping_[i];
    lowcmd_msg_.motor_cmd[joint_ids_map_[i]].dq = 0.0;
    lowcmd_msg_.motor_cmd[joint_ids_map_[i]].tau = 0.0;
  }
}

} // namespace legged_rl_deploy
