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
  if (configNode_["ll_dt"]) {
    ll_dt_ = configNode_["ll_dt"].as<double>();
  }

  const auto& pnode = configNode_["policy"];
  const std::string backend    = pnode["backend"].as<std::string>("torch");
  const std::string model_path = pnode["model_path"].as<std::string>();

  input_dim_ = pnode["input_dim"].as<size_t>();
  output_dim_ = pnode["output_dim"].as<size_t>();

  policy_runner_ = makePolicyRunner(backend);
  policy_runner_->load(model_path, input_dim_, output_dim_);
  std::cout << "[LeggedRLDeploy] Policy load successfully." << std::endl;

  input_buf_.assign(input_dim_, 0.0f);
  output_buf_.assign(output_dim_, 0.0f);

  policy_dt_ = pnode["policy_dt"].as<float>(0.02f);
  joint_ids_map_ = pnode["joint_ids_map"].as<std::vector<size_t>>();
  stiffness_ = pnode["stiffness"].as<std::vector<float>>();
  damping_ = pnode["damping"].as<std::vector<float>>();
  last_action_.assign(damping_.size(), 0.0f);

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

  stack_len_ = pnode["observations"]["stack"]["length"].as<size_t>();
  stack_order_ = pnode["observations"]["stack"]["order"].as<std::string>();
  registryObsTerms(pnode["observations"]["terms"]);

  std::cout << "[LeggedRLDeploy] init done. input_dim=" << input_dim_
            << " output_dim=" << output_dim_ << std::endl;
}

void LeggedRLDeploy::resetHighController() {
  std::fill(input_buf_.begin(), input_buf_.end(), 0.0f);
  std::fill(output_buf_.begin(), output_buf_.end(), 0.0f);
  std::fill(last_action_.begin(), last_action_.end(), 0.0f);
  obs_hist_.clear();
}

void LeggedRLDeploy::registryObsTerms(YAML::Node node) {
  obs_terms_.clear();
  size_t off = 0;

  if (!node || !node.IsMap()) {
    throw std::runtime_error("[LeggedRLDeploy] policy.observations.terms missing or not a map");
  }

  for (auto it = node.begin(); it != node.end(); ++it) {
    ObsTerm t;
    t.name = it->first.as<std::string>();
    YAML::Node term_node = it->second;
    t.params = term_node["params"];

    // -------- dim inference --------
    calculateObsTerm(t);

    t.offset = off;
    off += t.dim;

    auto maybe = Processor::TryLoad(term_node);
    if (maybe) t.proc = std::move(*maybe);

    obs_terms_.push_back(std::move(t));
  }

  obs_dim_ = off;

  const size_t expected = obs_dim_ * stack_len_;
  if (expected != input_dim_) {
    throw std::runtime_error("[LeggedRLDeploy] input_dim mismatch: obs_dim=" +
                             std::to_string(obs_dim_) + " stack_len=" +
                             std::to_string(stack_len_) + " expected=" +
                             std::to_string(expected) + " cfg input_dim=" +
                             std::to_string(input_dim_));
  }

  obs_now_.assign(obs_dim_, 0.0f);
  obs_hist_.clear();
}

void LeggedRLDeploy::assembleObsFrame() {
  std::fill(obs_now_.begin(), obs_now_.end(), 0.0f);

  for (const auto& t : obs_terms_) {
    std::vector<float> v(t.dim, 0.0f);
    if (t.name == "constants") {
      v = t.params["vec"].as<std::vector<float>>();
    } else if (t.name == "joystick_buttons") {
      const auto keys = t.params["keys"].as<std::vector<std::string>>();
      for (size_t i = 0; i < keys.size(); ++i) {
        unitree::common::Button btn;
        if (keys[i] == "A") btn = gamepad_.A;
        else if (keys[i] == "B") btn = gamepad_.B;
        else if (keys[i] == "X") btn = gamepad_.X;
        else if (keys[i] == "Y") btn = gamepad_.Y;
        else if (keys[i] == "up") btn = gamepad_.up;
        else if (keys[i] == "down") btn = gamepad_.down;
        else if (keys[i] == "left") btn = gamepad_.left;
        else if (keys[i] == "right") btn = gamepad_.right;
        else if (keys[i] == "L1") btn = gamepad_.L1;
        else if (keys[i] == "L2") btn = gamepad_.L2;
        else if (keys[i] == "R1") btn = gamepad_.R1;
        else if (keys[i] == "R2") btn = gamepad_.R2;
        else if (keys[i] == "start") btn = gamepad_.start;
        else if (keys[i] == "select") btn = gamepad_.select;
        else {
          throw std::runtime_error("[LeggedRLDeploy] Unknown joystick button: " + keys[i]);
        }
        v[i] = btn.pressed ? 1.0f : 0.0f;
      }
      // std::cout << "[LeggedRLDeploy] joystick_buttons : ";
      // for (size_t i = 0; i < t.dim; ++i) {
      //   std::cout << v[i] << " ";
      // }
      // std::cout << std::endl;
    } else if (t.name == "gait_phase_2") {
      float cycle_time = t.params["cycle_time"].as<float>();
      const float phase = loop_cnt_ * ll_dt_ / cycle_time;
      v[0] = sinf(2.0f * M_PI * phase);
      v[1] = cosf(2.0f * M_PI * phase);
    } else if (t.name == "base_ang_vel_W") {
      v[0] = real_state_.base_ang_vel_W()[0];
      v[1] = real_state_.base_ang_vel_W()[1];
      v[2] = real_state_.base_ang_vel_W()[2];
    } else if (t.name == "base_ang_vel_B") {
      v[0] = real_state_.base_ang_vel_B()[0];
      v[1] = real_state_.base_ang_vel_B()[1];
      v[2] = real_state_.base_ang_vel_B()[2];
    } else if (t.name == "eulerZYX_rpy") {
      // v[0] = real_state_.base_eulerZYX()[2];
      // v[1] = real_state_.base_eulerZYX()[1];
      // v[2] = real_state_.base_eulerZYX()[0];
      auto quat_to_rpy = [](const Eigen::Quaterniond& q_in) -> std::vector<float> {
          // Python 里 quat = [x, y, z, w]
          const double x = q_in.x();
          const double y = q_in.y();
          const double z = q_in.z();
          const double w = q_in.w();

          // roll (x)
          const double t0 = 2.0 * (w * x + y * z);
          const double t1 = 1.0 - 2.0 * (x * x + y * y);
          double roll = std::atan2(t0, t1);

          // pitch (y)
          double t2 = 2.0 * (w * y - z * x);
          t2 = std::clamp(t2, -1.0, 1.0);
          double pitch = std::asin(t2);

          // yaw (z)
          const double t3 = 2.0 * (w * z + x * y);
          const double t4 = 1.0 - 2.0 * (y * y + z * z);
          double yaw = std::atan2(t3, t4);

          std::vector<float> rpy = {(float)roll, (float)pitch, (float)yaw};

          // 对齐 Python:
          // eu_ang[eu_ang > pi] -= 2*pi
          for (int i = 0; i < 3; ++i) {
              if (rpy[i] > M_PI) {
                  rpy[i] -= 2.0 * M_PI;
              }
          }

          return rpy;
      };
      v = quat_to_rpy(real_state_.base_quat());
    } else if (t.name == "projected_gravity") {
      Eigen::Vector3d g = real_state_.base_quat().conjugate() * Eigen::Vector3d(0,0,-1);
      v[0] = (float)g[0]; v[1] = (float)g[1]; v[2] = (float)g[2];
    } else if (t.name == "velocity_commands") {
      v = {gamepad_.ly, -gamepad_.lx, -gamepad_.rx};
      auto it = commands_.find("base_velocity");
      if (it != commands_.end()) it->second.process(v);
    } else if (t.name == "joint_pos") {
      for (size_t i = 0; i < robot_model_.nJoints(); ++i)
        v[i] = lowstate_msg_.motor_state[joint_ids_map_[i]].q;
    } else if (t.name == "joint_vel") {
      for (size_t i = 0; i < robot_model_.nJoints(); ++i)
        v[i] = lowstate_msg_.motor_state[joint_ids_map_[i]].dq;
    } else if (t.name == "last_action") {
      v = last_action_;
    } else {
      throw std::runtime_error("Unknown obs term: " + t.name);
    }

    if (t.proc) t.proc->process(v);
    std::copy(v.begin(), v.end(), obs_now_.begin() + t.offset);
  }
}

void LeggedRLDeploy::stackObsGlobal() {
  // push newest to back, keep length
  obs_hist_.push_back(obs_now_);
  while (obs_hist_.size() > stack_len_) obs_hist_.pop_front();

  // warm start: 没满就用最旧的复制补齐
  while (obs_hist_.size() < stack_len_) obs_hist_.push_front(obs_hist_.front());

  // flatten
  size_t out = 0;
  if (stack_order_ == "newest_first") {
    for (size_t k = 0; k < stack_len_; ++k) {
      const auto& fr = obs_hist_[stack_len_ - 1 - k];  // newest -> oldest
      std::copy(fr.begin(), fr.end(), input_buf_.begin() + out);
      out += obs_dim_;
    }
  } else {
    for (size_t k = 0; k < stack_len_; ++k) {
      const auto& fr = obs_hist_[k];                   // oldest -> newest
      std::copy(fr.begin(), fr.end(), input_buf_.begin() + out);
      out += obs_dim_;
    }
  }
}

void LeggedRLDeploy::updateHighController() {
  const int decim = std::max(1, (int)std::lround(policy_dt_ / ll_dt_));
  if ((loop_cnt_ % decim) != 0) {
    return;
  }

  // 1) build current frame
  assembleObsFrame();

  // 2) stack to input_buf_
  if (stack_len_ == 1) {
    std::copy(obs_now_.begin(), obs_now_.end(), input_buf_.begin());
  } else {
    stackObsGlobal();
  }

  // 3) infer
  policy_runner_->infer(input_buf_.data(), output_buf_.data());

  // 4) last_action must be RAW
  last_action_ = output_buf_;

  // 5) action postprocess
  auto it = actions_.find("JointPositionAction");
  if (it != actions_.end()) it->second.process(output_buf_);

  // 6) send lowcmd (remember dq/tau!)
  for (size_t i = 0; i < output_dim_; ++i) {
    const size_t j = joint_ids_map_[i];
    lowcmd_msg_.motor_cmd[j].q   = output_buf_[i];
    lowcmd_msg_.motor_cmd[j].kp  = stiffness_[i];
    lowcmd_msg_.motor_cmd[j].kd  = damping_[i];
    lowcmd_msg_.motor_cmd[j].dq  = 0.0f;
    lowcmd_msg_.motor_cmd[j].tau = 0.0f;
  }
}

} // namespace legged_rl_deploy
