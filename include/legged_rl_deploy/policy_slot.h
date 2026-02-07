#pragma once

#include <cmath>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include "legged_rl_deploy/motion/motion_loader.h"
#include "legged_rl_deploy/policy/i_policy_runner.h"
#include "legged_rl_deploy/processor.h"

#include <legged_base/LeggedModel.h>
#include <legged_base/LeggedState.h>
#include <unitree_lowlevel/gamepad.hpp>

namespace legged_rl_deploy {

// ---------------------------------------------------------------------------
// PolicySlot — 一个完全自包含的 policy 实例
//
// 内含：model runner、input/output buf、obs pipeline、action pipeline、motion
// 不依赖 LeggedRLDeploy 的任何成员（通过方法参数传入共享的只读状态）。
// ---------------------------------------------------------------------------
class PolicySlot {
public:
  PolicySlot(const std::string& name, const YAML::Node& policyNode,
             const LeggedModel& model);

  // -------- lifecycle --------
  void init();
  void reset(const LeggedState& state);
  void update(const LeggedState& state,
              const unitree::common::Gamepad& gamepad,
              size_t loop_cnt, double ll_dt);

  // -------- accessors (const) --------
  const std::string& name() const { return name_; }
  float policyDt() const { return policy_dt_; }
  size_t outputDim() const { return output_dim_; }
  const std::vector<float>& outputBuf() const { return output_buf_; }
  const std::vector<size_t>& jointIdsMap() const { return joint_ids_map_; }
  const std::vector<float>& stiffness() const { return stiffness_; }
  const std::vector<float>& damping() const { return damping_; }

private:
  // -------- obs helpers (moved from LeggedRLDeploy) --------
  void registryObsTerms(YAML::Node node);
  void assembleObsFrame(const LeggedState& state,
                        const unitree::common::Gamepad& gamepad,
                        size_t loop_cnt, double ll_dt);
  void stackObsGlobal();
  void updatePolicy(const LeggedState& state,
                    const unitree::common::Gamepad& gamepad,
                    size_t loop_cnt, double ll_dt);

  // -------- ObsTerm (same structure as before) --------
  struct ObsTerm {
    std::string name;
    size_t dim = 0;
    size_t offset = 0;
    std::optional<Processor> proc;
    YAML::Node params;
  };
  void calculateObsTerm(ObsTerm& term);

  // -------- identity --------
  std::string name_;
  YAML::Node policyNode_;
  const LeggedModel& robot_model_;   // borrowed reference (shared, read-only)

  // -------- policy runner --------
  std::unique_ptr<IPolicyRunner> policy_runner_;
  size_t input_dim_ = 0, output_dim_ = 0;
  std::vector<float> input_buf_, output_buf_;

  // -------- control params --------
  float policy_dt_ = 0.02f;
  std::vector<size_t> joint_ids_map_;
  std::vector<float> stiffness_, damping_, last_action_;

  // -------- processors --------
  std::unordered_map<std::string, Processor> commands_, actions_;

  // -------- observations --------
  std::vector<ObsTerm> obs_terms_;
  size_t stack_len_ = 1;
  std::string stack_order_ = "oldest_first";
  size_t obs_dim_ = 0;
  std::vector<float> obs_now_;
  std::deque<std::vector<float>> obs_hist_;

  // -------- mimic motion --------
  std::unique_ptr<MotionLoader> motion_;
  size_t motion_cnt_ = 0;
  float motion_time_start_ = 0.0f;
  Eigen::Quaternionf motion_yaw_align_ = Eigen::Quaternionf::Identity();
};

} // namespace legged_rl_deploy
