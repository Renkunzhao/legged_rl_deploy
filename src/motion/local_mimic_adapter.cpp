#include "legged_rl_deploy/motion/local_mimic_adapter.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "legged_rl_deploy/motion/cnpy.h"
#include <legged_base/math/eigen_utils.hpp>
#include <legged_base/math/rotation_euler_zyx.hpp>
#include <unitree_lowlevel/adapter/g1_adapter.hpp>

namespace legged_rl_deploy {

LocalMimicAdapter::LocalMimicAdapter(Config cfg, const LeggedModel& robot_model,
                                     std::vector<size_t> joint_ids_map,
                                     size_t output_dim)
    : cfg_(std::move(cfg)),
      robot_model_(robot_model),
      joint_ids_map_(std::move(joint_ids_map)),
      output_dim_(output_dim) {
  if (cfg_.file.empty()) {
    throw std::runtime_error("[LocalMimicAdapter] local.file is required");
  }
  if (cfg_.fps <= 0.0f) {
    throw std::runtime_error("[LocalMimicAdapter] local.fps must be positive");
  }

  motion_dt_ = 1.0f / cfg_.fps;

  if (endsWith(cfg_.file, ".csv")) {
    loadDataFromCsv(cfg_.file);
  } else if (endsWith(cfg_.file, ".onnx")) {
    onnx_eval_ = makeOnnxMotionEvaluator(cfg_.file);
  } else {
    loadDataFromNpz(cfg_.file);
  }

  if (onnx_eval_) {
    num_frames_ = 0;
    duration_ = (cfg_.time_end > 0.0f) ? cfg_.time_end : 1e6f;
    std::cout << "[LocalMimicAdapter] ONNX on-demand mode" << std::endl;
  } else {
    num_frames_ = static_cast<int>(dof_positions_.size());
    duration_ = num_frames_ * motion_dt_;
  }

  time_start_ = std::clamp(cfg_.time_start, 0.0f, duration_);
  time_end_ = (cfg_.time_end < 0.0f) ? duration_
                                     : std::clamp(cfg_.time_end, 0.0f, duration_);

  const size_t inferred = inferOutputDim();
  if (output_dim_ != inferred) {
    throw std::runtime_error(
        "[LocalMimicAdapter] dim mismatch: inferred=" + std::to_string(inferred) +
        " cfg dim=" + std::to_string(output_dim_));
  }

  update(time_start_);
  last_output_.assign(output_dim_, 0.0f);
}

void LocalMimicAdapter::reset(const LeggedState& state, float policy_dt) {
  policy_dt_ = policy_dt;
  step_cnt_ = 0;
  yaw_align_ = Eigen::Quaternionf::Identity();

  update(time_start_);

  const Eigen::VectorXf ref_jp_urdf = jointPosUrdfOrder();
  const auto ref_q = G1Adapter::getTorsoQuatFromImuAndWaist(rootQuaternion(), ref_jp_urdf);
  const auto real_q = G1Adapter::getTorsoQuatFromImuAndWaist(
      state.base_quat().cast<float>(), state.joint_pos().cast<float>());

  const Eigen::Matrix3f ref_yaw =
      legged_base::extractYawQuaternion(ref_q).toRotationMatrix();
  const Eigen::Matrix3f real_yaw =
      legged_base::extractYawQuaternion(real_q).toRotationMatrix();
  yaw_align_ = Eigen::Quaternionf(real_yaw * ref_yaw.transpose());

  buildOutput(state);
}

void LocalMimicAdapter::step(const LeggedState& state) {
  step_cnt_++;
  const float t = step_cnt_ * policy_dt_ + time_start_;
  update(t);
  buildOutput(state);
}

void LocalMimicAdapter::read(std::vector<float>& out) { out = last_output_; }

void LocalMimicAdapter::loadDataFromNpz(const std::string& motion_file) {
  cnpy::npz_t npz_data = cnpy::npz_load(motion_file);

  auto body_pos_w = npz_data["body_pos_w"];
  auto body_quat_w = npz_data["body_quat_w"];
  auto joint_pos = npz_data["joint_pos"];
  auto joint_vel = npz_data["joint_vel"];

  root_positions_.clear();
  root_quaternions_.clear();
  dof_positions_.clear();
  dof_velocities_.clear();

  const size_t num_frames_npz = body_pos_w.shape[0];
  for (size_t i = 0; i < num_frames_npz; i++) {
    const size_t body_stride_pos = body_pos_w.shape[1] * body_pos_w.shape[2];
    const size_t body_stride_quat = body_quat_w.shape[1] * body_quat_w.shape[2];

    root_positions_.push_back(
        Eigen::Vector3f::Map(body_pos_w.data<float>() + i * body_stride_pos));

    Eigen::Quaternionf quat(body_quat_w.data<float>()[i * body_stride_quat + 0],
                            body_quat_w.data<float>()[i * body_stride_quat + 1],
                            body_quat_w.data<float>()[i * body_stride_quat + 2],
                            body_quat_w.data<float>()[i * body_stride_quat + 3]);
    root_quaternions_.push_back(quat);

    Eigen::VectorXf joint_position(joint_pos.shape[1]);
    for (int j = 0; j < joint_pos.shape[1]; j++) {
      joint_position[j] = joint_pos.data<float>()[i * joint_pos.shape[1] + j];
    }

    Eigen::VectorXf joint_velocity(joint_vel.shape[1]);
    for (int j = 0; j < joint_vel.shape[1]; j++) {
      joint_velocity[j] = joint_vel.data<float>()[i * joint_vel.shape[1] + j];
    }

    dof_positions_.push_back(joint_position);
    dof_velocities_.push_back(joint_velocity);
  }
}

void LocalMimicAdapter::loadDataFromCsv(const std::string& motion_file) {
  auto data = loadCsv(motion_file);
  if (data.empty()) {
    throw std::runtime_error("[LocalMimicAdapter] CSV motion is empty: " + motion_file);
  }

  root_positions_.clear();
  root_quaternions_.clear();
  dof_positions_.clear();
  dof_velocities_.clear();

  for (size_t i = 0; i < data.size(); ++i) {
    if (data[i].size() < 7) {
      throw std::runtime_error("[LocalMimicAdapter] CSV row " + std::to_string(i) +
                               " has insufficient columns");
    }
    root_positions_.push_back(Eigen::Vector3f(data[i][0], data[i][1], data[i][2]));
    root_quaternions_.push_back(
        Eigen::Quaternionf(data[i][6], data[i][3], data[i][4], data[i][5]));
    dof_positions_.push_back(
        Eigen::VectorXf::Map(data[i].data() + 7, data[i].size() - 7));
  }

  dof_velocities_ = computeRawDerivative(dof_positions_);
}

void LocalMimicAdapter::update(float time) {
  if (onnx_eval_) {
    const float time_step = std::max(0.0f, time) / motion_dt_;
    onnx_eval_->evaluate(time_step, onnx_joint_pos_, onnx_joint_vel_, onnx_root_quat_);
    return;
  }

  const float phase = std::clamp(time, 0.0f, duration_);
  const float f = phase / motion_dt_;
  frame_ = static_cast<int>(std::floor(f));
  frame_ = std::min(frame_, std::max(0, num_frames_ - 1));
}

void LocalMimicAdapter::buildOutput(const LeggedState& state) {
  last_output_.clear();
  last_output_.reserve(output_dim_);

  for (const auto& term : cfg_.terms) {
    if (term == "joint_pos") {
      const Eigen::VectorXf jp = jointPosTrainingOrder();
      for (int i = 0; i < jp.size(); ++i) {
        last_output_.push_back(jp[i]);
      }
      continue;
    }

    if (term == "joint_vel") {
      const Eigen::VectorXf jv = jointVelTrainingOrder();
      for (int i = 0; i < jv.size(); ++i) {
        last_output_.push_back(jv[i]);
      }
      continue;
    }

    if (term == "motion_anchor_ori_b") {
      const Eigen::VectorXf ref_jp = jointPosUrdfOrder();
      const auto ref_q = G1Adapter::getTorsoQuatFromImuAndWaist(rootQuaternion(), ref_jp);
      const auto real_q = G1Adapter::getTorsoQuatFromImuAndWaist(
          state.base_quat().cast<float>(), state.joint_pos().cast<float>());
      const auto rot_ = (yaw_align_ * ref_q).conjugate() * real_q;
      const Eigen::Matrix3f rot = rot_.toRotationMatrix().transpose();

      last_output_.push_back(rot(0, 0));
      last_output_.push_back(rot(0, 1));
      last_output_.push_back(rot(1, 0));
      last_output_.push_back(rot(1, 1));
      last_output_.push_back(rot(2, 0));
      last_output_.push_back(rot(2, 1));
      continue;
    }

    throw std::runtime_error("[LocalMimicAdapter] Unsupported mimic term: " + term);
  }

  if (last_output_.size() != output_dim_) {
    throw std::runtime_error("[LocalMimicAdapter] Built dim mismatch: got " +
                             std::to_string(last_output_.size()) + " expected " +
                             std::to_string(output_dim_));
  }
}

Eigen::VectorXf LocalMimicAdapter::jointPosRaw() const {
  return onnx_eval_ ? onnx_joint_pos_ : dof_positions_[frame_];
}

Eigen::VectorXf LocalMimicAdapter::jointVelRaw() const {
  return onnx_eval_ ? onnx_joint_vel_ : dof_velocities_[frame_];
}

Eigen::Quaternionf LocalMimicAdapter::rootQuaternion() const {
  return onnx_eval_ ? onnx_root_quat_ : root_quaternions_[frame_];
}

Eigen::VectorXf LocalMimicAdapter::jointPosTrainingOrder() const {
  const Eigen::VectorXf src = jointPosRaw();
  if (!cfg_.hardware_order) {
    return src;
  }

  Eigen::VectorXf out(src.size());
  for (size_t i = 0; i < joint_ids_map_.size(); ++i) {
    out[i] = src[joint_ids_map_[i]];
  }
  return out;
}

Eigen::VectorXf LocalMimicAdapter::jointVelTrainingOrder() const {
  const Eigen::VectorXf src = jointVelRaw();
  if (!cfg_.hardware_order) {
    return src;
  }

  Eigen::VectorXf out(src.size());
  for (size_t i = 0; i < joint_ids_map_.size(); ++i) {
    out[i] = src[joint_ids_map_[i]];
  }
  return out;
}

Eigen::VectorXf LocalMimicAdapter::jointPosUrdfOrder() const {
  const Eigen::VectorXf src = jointPosRaw();
  if (cfg_.hardware_order) {
    return src;
  }

  Eigen::VectorXf urdf(src.size());
  for (size_t i = 0; i < joint_ids_map_.size(); ++i) {
    urdf[joint_ids_map_[i]] = src[i];
  }
  return urdf;
}

bool LocalMimicAdapter::endsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::vector<float>> LocalMimicAdapter::loadCsv(
    const std::string& filename) {
  std::vector<std::vector<float>> data;
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("[LocalMimicAdapter] Error opening file: " + filename);
  }

  std::string line;
  while (std::getline(file, line)) {
    std::vector<float> row;
    std::stringstream ss(line);
    std::string value;
    while (std::getline(ss, value, ',')) {
      row.push_back(std::stof(value));
    }
    if (!row.empty()) {
      data.push_back(row);
    }
  }
  return data;
}

std::vector<Eigen::VectorXf> LocalMimicAdapter::computeRawDerivative(
    const std::vector<Eigen::VectorXf>& data) const {
  std::vector<Eigen::VectorXf> derivative;
  if (data.empty()) {
    return derivative;
  }

  for (size_t i = 0; i + 1 < data.size(); ++i) {
    derivative.push_back((data[i + 1] - data[i]) / motion_dt_);
  }
  derivative.push_back(derivative.empty() ? Eigen::VectorXf::Zero(data[0].size())
                                          : derivative.back());
  return derivative;
}

size_t LocalMimicAdapter::inferOutputDim() const {
  size_t dim = 0;
  for (const auto& term : cfg_.terms) {
    if (term == "joint_pos" || term == "joint_vel") {
      dim += robot_model_.nJoints();
      continue;
    }
    if (term == "motion_anchor_ori_b") {
      dim += 6;
      continue;
    }
    throw std::runtime_error("[LocalMimicAdapter] Unsupported mimic term: " + term);
  }
  return dim;
}

} // namespace legged_rl_deploy
