#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "legged_rl_deploy/deploy_config.h"

namespace legged_rl_deploy {

/**
 * JointPositionActionProcessor
 *
 * Converts policy output action (typically normalized, e.g. [-1,1]) into joint position targets:
 *   u = clip(policy_out, clip_low/high)          (elementwise)
 *   q_des = offset + scale * u                  (elementwise)
 *
 * Then maps q_des into robot joint order via joint_ids_map.
 *
 * Notes:
 * - This class assumes you have a JointPositionActionConfig loaded from YAML:
 *     actions:
 *       JointPositionAction:
 *         clip: [[low,high]...]
 *         scale: [...]
 *         offset: [...]
 *         joint_ids: null | [...]
 * - If joint_ids is provided in action config, it defines which joints the action vector controls.
 *   Otherwise, it defaults to [0..act_dim-1].
 * - joint_ids_map (size = robot_dof) maps policy/internal joint order -> robot SDK joint order.
 *   Example in your YAML: joint_ids_map: [3,0,9,6,4,1,10,7,5,2,11,8]
 */
class JointPositionActionProcessor {
public:
  // expected_act_dim: if >=0, validate policy action dim equals it
  JointPositionActionProcessor(const JointPositionActionConfig& cfg,
                               const std::vector<int>& joint_ids_map,
                               int expected_act_dim = -1)
      : cfg_(cfg), joint_ids_map_(joint_ids_map) {
    ValidateAndBuild(expected_act_dim);
  }

  int act_dim() const { return act_dim_; }                 // policy action dim (input)
  int robot_dof() const { return robot_dof_; }             // output size (per robot joint)
  const std::vector<int>& controlled_joint_ids() const { return controlled_joint_ids_; }

  /**
   * @brief Convert policy action -> robot joint position targets in robot joint index order.
   *
   * @param policy_action  size == act_dim()
   * @return q_des_robot   size == robot_dof()
   */
  std::vector<float> ComputeJointTargetsRobotOrder(const std::vector<float>& policy_action) const {
    if (static_cast<int>(policy_action.size()) != act_dim_) {
      throw std::runtime_error("ComputeJointTargetsRobotOrder(): policy_action dim mismatch: "
                               "expected " + std::to_string(act_dim_) + ", got " +
                               std::to_string(policy_action.size()));
    }

    // Step 1: build q_des in "internal joint index" space (0..robot_dof-1), initialize with NaN
    std::vector<float> q_des_internal(static_cast<size_t>(robot_dof_),
                                      std::numeric_limits<float>::quiet_NaN());

    // For each controlled joint, compute q_des = offset + scale * clipped_action
    for (int i = 0; i < act_dim_; ++i) {
      const int j_internal = controlled_joint_ids_[static_cast<size_t>(i)];

      float u = policy_action[static_cast<size_t>(i)];
      u = std::clamp(u, clip_low_[static_cast<size_t>(i)], clip_high_[static_cast<size_t>(i)]);

      const float q = offset_[static_cast<size_t>(i)] + scale_[static_cast<size_t>(i)] * u;
      q_des_internal[static_cast<size_t>(j_internal)] = q;
    }

    // Optional: fill uncontrolled joints with their default offset? Here we keep NaN unless you want.
    // In practice you often want to fill with default_joint_pos; do that outside using your DeployConfig.
    // We'll provide a helper overload below.

    // Step 2: map internal joint indices -> robot joint indices using joint_ids_map
    // joint_ids_map_[internal_idx] = robot_idx
    std::vector<float> q_des_robot(static_cast<size_t>(robot_dof_),
                                   std::numeric_limits<float>::quiet_NaN());

    for (int j_internal = 0; j_internal < robot_dof_; ++j_internal) {
      const int j_robot = joint_ids_map_[static_cast<size_t>(j_internal)];
      q_des_robot[static_cast<size_t>(j_robot)] = q_des_internal[static_cast<size_t>(j_internal)];
    }

    return q_des_robot;
  }

  /**
   * @brief Same as above, but fill uncontrolled joints with provided fallback (e.g. default_joint_pos).
   *
   * @param policy_action    size == act_dim()
   * @param fallback_internal size == robot_dof(), interpreted in internal joint index order
   * @return q_des_robot      size == robot_dof(), robot joint index order
   */
  std::vector<float> ComputeJointTargetsRobotOrder(const std::vector<float>& policy_action,
                                                   const std::vector<float>& fallback_internal) const {
    if (static_cast<int>(fallback_internal.size()) != robot_dof_) {
      throw std::runtime_error("ComputeJointTargetsRobotOrder(fallback): fallback_internal dim mismatch: "
                               "expected " + std::to_string(robot_dof_) + ", got " +
                               std::to_string(fallback_internal.size()));
    }

    if (static_cast<int>(policy_action.size()) != act_dim_) {
      throw std::runtime_error("ComputeJointTargetsRobotOrder(fallback): policy_action dim mismatch: "
                               "expected " + std::to_string(act_dim_) + ", got " +
                               std::to_string(policy_action.size()));
    }

    // internal joint targets start from fallback
    std::vector<float> q_des_internal = fallback_internal;

    for (int i = 0; i < act_dim_; ++i) {
      const int j_internal = controlled_joint_ids_[static_cast<size_t>(i)];
      float u = policy_action[static_cast<size_t>(i)];
      u = std::clamp(u, clip_low_[static_cast<size_t>(i)], clip_high_[static_cast<size_t>(i)]);
      q_des_internal[static_cast<size_t>(j_internal)] =
          offset_[static_cast<size_t>(i)] + scale_[static_cast<size_t>(i)] * u;
    }

    // map to robot order
    std::vector<float> q_des_robot(static_cast<size_t>(robot_dof_), 0.0f);
    for (int j_internal = 0; j_internal < robot_dof_; ++j_internal) {
      const int j_robot = joint_ids_map_[static_cast<size_t>(j_internal)];
      q_des_robot[static_cast<size_t>(j_robot)] = q_des_internal[static_cast<size_t>(j_internal)];
    }
    return q_des_robot;
  }

private:
  void ValidateAndBuild(int expected_act_dim) {
    // robot dof inferred from joint_ids_map size (e.g. 12)
    robot_dof_ = static_cast<int>(joint_ids_map_.size());
    if (robot_dof_ <= 0) {
      throw std::runtime_error("JointPositionActionProcessor: joint_ids_map is empty.");
    }

    // cfg vectors must be consistent
    act_dim_ = static_cast<int>(cfg_.scale.size());
    if (act_dim_ <= 0) {
      throw std::runtime_error("JointPositionActionProcessor: cfg.scale is empty.");
    }
    if (expected_act_dim >= 0 && act_dim_ != expected_act_dim) {
      throw std::runtime_error("JointPositionActionProcessor: action dim mismatch: expected " +
                               std::to_string(expected_act_dim) + ", got " +
                               std::to_string(act_dim_));
    }

    if (static_cast<int>(cfg_.offset.size()) != act_dim_) {
      throw std::runtime_error("JointPositionActionProcessor: cfg.offset size mismatch: "
                               "expected " + std::to_string(act_dim_) + ", got " +
                               std::to_string(cfg_.offset.size()));
    }
    if (static_cast<int>(cfg_.clip_low.size()) != act_dim_ ||
        static_cast<int>(cfg_.clip_high.size()) != act_dim_) {
      throw std::runtime_error("JointPositionActionProcessor: clip_low/high size mismatch: "
                               "expected " + std::to_string(act_dim_) + ", got " +
                               std::to_string(cfg_.clip_low.size()) + "/" +
                               std::to_string(cfg_.clip_high.size()));
    }

    // joint_ids_map must be a permutation of [0..robot_dof-1]
    {
      std::vector<int> seen(robot_dof_, 0);
      for (int v : joint_ids_map_) {
        if (v < 0 || v >= robot_dof_) {
          throw std::runtime_error("JointPositionActionProcessor: joint_ids_map contains out-of-range value " +
                                   std::to_string(v));
        }
        seen[static_cast<size_t>(v)]++;
      }
      for (int i = 0; i < robot_dof_; ++i) {
        if (seen[static_cast<size_t>(i)] != 1) {
          throw std::runtime_error("JointPositionActionProcessor: joint_ids_map is not a permutation "
                                   "(value " + std::to_string(i) + " appears " +
                                   std::to_string(seen[static_cast<size_t>(i)]) + " times).");
        }
      }
    }

    // controlled_joint_ids
    controlled_joint_ids_.clear();
    controlled_joint_ids_.reserve(static_cast<size_t>(act_dim_));

    if (cfg_.joint_ids.has_value()) {
      const auto& ids = cfg_.joint_ids.value();
      if (static_cast<int>(ids.size()) != act_dim_) {
        throw std::runtime_error("JointPositionActionProcessor: cfg.joint_ids size mismatch: "
                                 "expected " + std::to_string(act_dim_) + ", got " +
                                 std::to_string(ids.size()));
      }
      for (int j : ids) {
        if (j < 0 || j >= robot_dof_) {
          throw std::runtime_error("JointPositionActionProcessor: cfg.joint_ids contains out-of-range id " +
                                   std::to_string(j));
        }
        controlled_joint_ids_.push_back(j);
      }
    } else {
      // default: first act_dim joints in internal order
      if (act_dim_ != robot_dof_) {
        throw std::runtime_error("JointPositionActionProcessor: cfg.joint_ids is null, but act_dim (" +
                                 std::to_string(act_dim_) + ") != robot_dof (" +
                                 std::to_string(robot_dof_) + "). "
                                 "Either set joint_ids explicitly or make act_dim==robot_dof.");
      }
      for (int j = 0; j < act_dim_; ++j) controlled_joint_ids_.push_back(j);
    }

    // cache float vectors
    scale_.resize(static_cast<size_t>(act_dim_));
    offset_.resize(static_cast<size_t>(act_dim_));
    clip_low_.resize(static_cast<size_t>(act_dim_));
    clip_high_.resize(static_cast<size_t>(act_dim_));

    for (int i = 0; i < act_dim_; ++i) {
      scale_[static_cast<size_t>(i)] = static_cast<float>(cfg_.scale[static_cast<size_t>(i)]);
      offset_[static_cast<size_t>(i)] = static_cast<float>(cfg_.offset[static_cast<size_t>(i)]);
      clip_low_[static_cast<size_t>(i)] = static_cast<float>(cfg_.clip_low[static_cast<size_t>(i)]);
      clip_high_[static_cast<size_t>(i)] = static_cast<float>(cfg_.clip_high[static_cast<size_t>(i)]);
    }
  }

private:
  const JointPositionActionConfig& cfg_;
  const std::vector<int>& joint_ids_map_;

  int act_dim_ = 0;
  int robot_dof_ = 0;

  std::vector<int> controlled_joint_ids_;  // size act_dim

  // cached floats (size act_dim)
  std::vector<float> scale_;
  std::vector<float> offset_;
  std::vector<float> clip_low_;
  std::vector<float> clip_high_;
};

}  // namespace legged_rl_deploy
