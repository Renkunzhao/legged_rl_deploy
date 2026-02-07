#include "legged_rl_deploy/policy_slot.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "legged_rl_deploy/policy/policy_factory.h"
#include <legged_base/Utils.h>
#include <legged_base/math/eigen_utils.hpp>
#include <legged_base/math/rotation_euler_zyx.hpp>
#include <unitree_lowlevel/adapter/g1_adapter.hpp>

namespace legged_rl_deploy {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
PolicySlot::PolicySlot(const std::string& name, const YAML::Node& policyNode,
                       const LeggedModel& model)
    : name_(name), policyNode_(policyNode), robot_model_(model) {}

// ---------------------------------------------------------------------------
// init — load model, parse obs / action / motion (called once at startup)
// ---------------------------------------------------------------------------
void PolicySlot::init() {
  const auto& pnode = policyNode_;

  const std::string backend =
      pnode["backend"].as<std::string>("ort");
  const std::string model_path =
      legged_base::getEnv("WORKSPACE") + "/" +
      pnode["model_path"].as<std::string>();

  input_dim_ = pnode["input_dim"].as<size_t>();
  output_dim_ = pnode["output_dim"].as<size_t>();

  policy_runner_ = makePolicyRunner(backend);
  policy_runner_->load(model_path, input_dim_, output_dim_);
  std::cout << "[PolicySlot:" << name_
            << "] Policy loaded (" << backend << ")." << std::endl;

  input_buf_.assign(input_dim_, 0.0f);
  output_buf_.assign(output_dim_, 0.0f);

  policy_dt_ = pnode["policy_dt"].as<float>(0.02f);
  joint_ids_map_ = pnode["joint_ids_map"].as<std::vector<size_t>>();
  stiffness_ = pnode["stiffness"].as<std::vector<float>>();
  damping_ = pnode["damping"].as<std::vector<float>>();
  last_action_.assign(output_dim_, 0.0f);

  // -------- motion loader --------
  if (pnode["motions"]) {
    const std::string motion_file =
        legged_base::getEnv("WORKSPACE") + "/" +
        pnode["motions"]["file"].as<std::string>();
    const float motion_dt = 1.0f / pnode["motions"]["fps"].as<float>();
    const float time_start = pnode["motions"]["time_start"].as<float>(0.0f);
    const float time_end   = pnode["motions"]["time_end"].as<float>(-1.0f);
    motion_ = std::make_unique<MotionLoader>(
        motion_file, motion_dt, time_start, time_end);
    std::cout << "[PolicySlot:" << name_
              << "] Loaded motion: duration=" << motion_->duration
              << " dt=" << motion_->dt
              << " range=[" << motion_->timeStart()
              << ", " << motion_->timeEnd() << "]" << std::endl;
  }

  // -------- commands --------
  if (pnode["commands"]) {
    for (auto it = pnode["commands"].begin(); it != pnode["commands"].end();
         ++it) {
      const std::string cname = it->first.as<std::string>();
      auto maybe = Processor::TryLoad(it->second);
      if (maybe) commands_.emplace(cname, std::move(*maybe));
    }
  }

  // -------- actions --------
  if (pnode["actions"]) {
    for (auto it = pnode["actions"].begin(); it != pnode["actions"].end();
         ++it) {
      const std::string aname = it->first.as<std::string>();
      auto maybe = Processor::TryLoad(it->second);
      if (maybe) actions_.emplace(aname, std::move(*maybe));
    }
  }

  // -------- observations --------
  stack_len_ = pnode["observations"]["stack"]["length"].as<size_t>();
  stack_order_ = pnode["observations"]["stack"]["order"].as<std::string>();
  registryObsTerms(pnode["observations"]["terms"]);

  std::cout << "[PolicySlot:" << name_ << "] init done. input_dim="
            << input_dim_ << " output_dim=" << output_dim_ << std::endl;
}

// ---------------------------------------------------------------------------
// reset — clear buffers, re-align motion
// ---------------------------------------------------------------------------
void PolicySlot::reset(const LeggedState& state) {
  std::fill(input_buf_.begin(), input_buf_.end(), 0.0f);
  std::fill(output_buf_.begin(), output_buf_.end(), 0.0f);
  std::fill(last_action_.begin(), last_action_.end(), 0.0f);
  obs_hist_.clear();

  if (motion_) {
    motion_->reset(policy_dt_);

    const auto ref_q = G1Adapter::getTorsoQuatFromImuAndWaist(
        motion_->root_quaternion(), motion_->joint_pos());
    const auto real_q = G1Adapter::getTorsoQuatFromImuAndWaist(
        state.base_quat().cast<float>(),
        state.joint_pos().cast<float>());

    const Eigen::Matrix3f ref_yaw =
        legged_base::extractYawQuaternion(ref_q).toRotationMatrix();
    const Eigen::Matrix3f real_yaw =
        legged_base::extractYawQuaternion(real_q).toRotationMatrix();
    motion_->yawAlign() = Eigen::Quaternionf(real_yaw * ref_yaw.transpose());
  }

  std::cout << "[PolicySlot:" << name_ << "] reset." << std::endl;
}

// ---------------------------------------------------------------------------
// registryObsTerms
// ---------------------------------------------------------------------------
void PolicySlot::registryObsTerms(YAML::Node node) {
  obs_terms_.clear();
  size_t off = 0;

  if (!node || !node.IsMap()) {
    throw std::runtime_error("[PolicySlot:" + name_ +
                             "] observations.terms missing or not a map");
  }

  for (auto it = node.begin(); it != node.end(); ++it) {
    ObsTerm t;
    t.name = it->first.as<std::string>();
    YAML::Node term_node = it->second;
    t.params = term_node["params"];

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
    throw std::runtime_error(
        "[PolicySlot:" + name_ + "] input_dim mismatch: obs_dim=" +
        std::to_string(obs_dim_) + " stack_len=" +
        std::to_string(stack_len_) + " expected=" +
        std::to_string(expected) + " cfg input_dim=" +
        std::to_string(input_dim_));
  }

  obs_now_.assign(obs_dim_, 0.0f);
  obs_hist_.clear();
}

// ---------------------------------------------------------------------------
// calculateObsTerm — infer dim from name (unchanged logic)
// ---------------------------------------------------------------------------
void PolicySlot::calculateObsTerm(ObsTerm& term) {
  if (term.name == "constants") {
    if (!term.params || !term.params["vec"]) {
      throw std::runtime_error("[PolicySlot:" + name_ + "] " + term.name +
                               " requires params.vec");
    }
    term.dim = term.params["vec"].as<std::vector<float>>().size();
  }
  if (term.name == "joystick_buttons") {
    if (!term.params || !term.params["keys"]) {
      throw std::runtime_error("[PolicySlot:" + name_ + "] " + term.name +
                               " requires params.keys");
    }
    term.dim = term.params["keys"].as<std::vector<std::string>>().size();
  }
  if (term.name == "gait_phase_2")       term.dim = 2;
  if (term.name == "base_ang_vel_W")     term.dim = 3;
  if (term.name == "base_ang_vel_B")     term.dim = 3;
  if (term.name == "projected_gravity")  term.dim = 3;
  if (term.name == "eulerZYX_rpy")       term.dim = 3;
  if (term.name == "velocity_commands")  term.dim = 3;
  if (term.name == "joint_pos")          term.dim = robot_model_.nJoints();
  if (term.name == "joint_vel")          term.dim = robot_model_.nJoints();
  if (term.name == "last_action")        term.dim = output_dim_;
  if (term.name == "motion_command")     term.dim = 2 * robot_model_.nJoints();
  if (term.name == "motion_anchor_ori_b") term.dim = 6;
}

// ---------------------------------------------------------------------------
// assembleObsFrame
// ---------------------------------------------------------------------------
void PolicySlot::assembleObsFrame(const LeggedState& state,
                                  const unitree::common::Gamepad& gamepad,
                                  size_t loop_cnt, double ll_dt) {
  std::fill(obs_now_.begin(), obs_now_.end(), 0.0f);

  if (motion_) {
    motion_->step();
  }

  // Motion reference
  const bool has_motion = (motion_ != nullptr);

  for (const auto& term : obs_terms_) {
    std::vector<float> v(term.dim, 0.0f);

    if (term.name == "constants") {
      v = term.params["vec"].as<std::vector<float>>();

    } else if (term.name == "joystick_buttons") {
      const auto keys = term.params["keys"].as<std::vector<std::string>>();
      for (size_t i = 0; i < keys.size(); ++i) {
        unitree::common::Button btn;
        if      (keys[i] == "A")      btn = gamepad.A;
        else if (keys[i] == "B")      btn = gamepad.B;
        else if (keys[i] == "X")      btn = gamepad.X;
        else if (keys[i] == "Y")      btn = gamepad.Y;
        else if (keys[i] == "up")     btn = gamepad.up;
        else if (keys[i] == "down")   btn = gamepad.down;
        else if (keys[i] == "left")   btn = gamepad.left;
        else if (keys[i] == "right")  btn = gamepad.right;
        else if (keys[i] == "L1")     btn = gamepad.L1;
        else if (keys[i] == "L2")     btn = gamepad.L2;
        else if (keys[i] == "R1")     btn = gamepad.R1;
        else if (keys[i] == "R2")     btn = gamepad.R2;
        else if (keys[i] == "start")  btn = gamepad.start;
        else if (keys[i] == "select") btn = gamepad.select;
        else {
          throw std::runtime_error("[PolicySlot:" + name_ +
                                   "] Unknown joystick button: " + keys[i]);
        }
        v[i] = btn.pressed ? 1.0f : 0.0f;
      }

    } else if (term.name == "gait_phase_2") {
      float cycle_time = term.params["cycle_time"].as<float>();
      const float phase = loop_cnt * ll_dt / cycle_time;
      v[0] = sinf(2.0f * M_PI * phase);
      v[1] = cosf(2.0f * M_PI * phase);

    } else if (term.name == "base_ang_vel_W") {
      v[0] = state.base_ang_vel_W()[0];
      v[1] = state.base_ang_vel_W()[1];
      v[2] = state.base_ang_vel_W()[2];

    } else if (term.name == "base_ang_vel_B") {
      v[0] = state.base_ang_vel_B()[0];
      v[1] = state.base_ang_vel_B()[1];
      v[2] = state.base_ang_vel_B()[2];

    } else if (term.name == "eulerZYX_rpy") {
      auto quat_to_rpy = [](const Eigen::Quaterniond& q_in)
          -> std::vector<float> {
        const double x = q_in.x(), y = q_in.y(), z = q_in.z(), w = q_in.w();
        const double t0 = 2.0 * (w * x + y * z);
        const double t1 = 1.0 - 2.0 * (x * x + y * y);
        double roll = std::atan2(t0, t1);

        double t2 = 2.0 * (w * y - z * x);
        t2 = std::clamp(t2, -1.0, 1.0);
        double pitch = std::asin(t2);

        const double t3 = 2.0 * (w * z + x * y);
        const double t4 = 1.0 - 2.0 * (y * y + z * z);
        double yaw = std::atan2(t3, t4);

        std::vector<float> rpy = {(float)roll, (float)pitch, (float)yaw};
        for (int i = 0; i < 3; ++i) {
          if (rpy[i] > M_PI) rpy[i] -= 2.0 * M_PI;
        }
        return rpy;
      };
      v = quat_to_rpy(state.base_quat());

    } else if (term.name == "projected_gravity") {
      Eigen::Vector3d g =
          state.base_quat().conjugate() * Eigen::Vector3d(0, 0, -1);
      v[0] = (float)g[0]; v[1] = (float)g[1]; v[2] = (float)g[2];

    } else if (term.name == "velocity_commands") {
      v = {gamepad.ly, -gamepad.lx, -gamepad.rx};
      auto it = commands_.find("base_velocity");
      if (it != commands_.end()) it->second.process(v);

    } else if (term.name == "joint_pos") {
      for (size_t i = 0; i < robot_model_.nJoints(); ++i)
        v[i] = state.joint_pos()[joint_ids_map_[i]];

    } else if (term.name == "joint_vel") {
      for (size_t i = 0; i < robot_model_.nJoints(); ++i)
        v[i] = state.joint_vel()[joint_ids_map_[i]];

    } else if (term.name == "last_action") {
      v = last_action_;

    } else if (term.name == "motion_command") {
      if (!has_motion)
        throw std::runtime_error("[PolicySlot:" + name_ +
                                 "] motion_command requires motion");
      for (size_t i = 0; i < robot_model_.nJoints(); ++i) {
        v[i] = motion_->joint_pos()[joint_ids_map_[i]];
        v[i + robot_model_.nJoints()] =
            motion_->joint_vel()[joint_ids_map_[i]];
      }

    } else if (term.name == "motion_anchor_ori_b") {
      if (!has_motion)
        throw std::runtime_error("[PolicySlot:" + name_ +
                                 "] motion_anchor_ori_b requires motion");
      const auto ref_q = G1Adapter::getTorsoQuatFromImuAndWaist(
          motion_->root_quaternion(), motion_->joint_pos());
      const auto real_q = G1Adapter::getTorsoQuatFromImuAndWaist(
          state.base_quat().cast<float>(),
          state.joint_pos().cast<float>());
      const auto rot_ = (motion_->yawAlign() * ref_q).conjugate() * real_q;
      const Eigen::Matrix3f rot = rot_.toRotationMatrix().transpose();
      v[0] = rot(0, 0); v[1] = rot(0, 1);
      v[2] = rot(1, 0); v[3] = rot(1, 1);
      v[4] = rot(2, 0); v[5] = rot(2, 1);

    } else {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] Unknown obs term: " + term.name);
    }

    if (term.proc) term.proc->process(v);
    std::copy(v.begin(), v.end(), obs_now_.begin() + term.offset);
  }
}

// ---------------------------------------------------------------------------
// stackObsGlobal
// ---------------------------------------------------------------------------
void PolicySlot::stackObsGlobal() {
  obs_hist_.push_back(obs_now_);
  while (obs_hist_.size() > stack_len_) obs_hist_.pop_front();
  while (obs_hist_.size() < stack_len_) obs_hist_.push_front(obs_hist_.front());

  size_t out = 0;
  if (stack_order_ == "newest_first") {
    for (size_t k = 0; k < stack_len_; ++k) {
      const auto& fr = obs_hist_[stack_len_ - 1 - k];
      std::copy(fr.begin(), fr.end(), input_buf_.begin() + out);
      out += obs_dim_;
    }
  } else {
    for (size_t k = 0; k < stack_len_; ++k) {
      const auto& fr = obs_hist_[k];
      std::copy(fr.begin(), fr.end(), input_buf_.begin() + out);
      out += obs_dim_;
    }
  }
}

// ---------------------------------------------------------------------------
// updatePolicy — assemble obs → stack → infer → postprocess
// ---------------------------------------------------------------------------
void PolicySlot::updatePolicy(const LeggedState& state,
                              const unitree::common::Gamepad& gamepad,
                              size_t loop_cnt, double ll_dt) {
  assembleObsFrame(state, gamepad, loop_cnt, ll_dt);

  if (stack_len_ == 1) {
    std::copy(obs_now_.begin(), obs_now_.end(), input_buf_.begin());
  } else {
    stackObsGlobal();
  }

  policy_runner_->infer(input_buf_.data(), output_buf_.data());

  // last_action must be RAW (before action postprocess)
  last_action_ = output_buf_;

  auto it = actions_.find("JointPositionAction");
  if (it != actions_.end()) it->second.process(output_buf_);
}

// ---------------------------------------------------------------------------
// update — decimation + updatePolicy
// ---------------------------------------------------------------------------
void PolicySlot::update(const LeggedState& state,
                        const unitree::common::Gamepad& gamepad,
                        size_t loop_cnt, double ll_dt) {
  const int decim = std::max(1, (int)std::lround(policy_dt_ / ll_dt));
  if ((loop_cnt % decim) == 0) {
    updatePolicy(state, gamepad, loop_cnt, ll_dt);
  }
}

} // namespace legged_rl_deploy
