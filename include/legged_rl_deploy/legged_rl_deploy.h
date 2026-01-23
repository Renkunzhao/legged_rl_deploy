#pragma once

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <deque>

#include "legged_rl_deploy/policy/i_policy_runner.h"
#include "legged_rl_deploy/processor.h"

#include <unitree_lowlevel/lowlevel_controller.h>
#include <legged_model/LeggedModel.h>

namespace legged_rl_deploy {

class LeggedRLDeploy : public LowLevelController {
public:
  LeggedRLDeploy(std::string configFile);

private:
//   void updateBaseState() override;
  void initHighController() override;
  void resetHighController() override;
  void updateHighController() override;
//   void log() override;

  void registryObsTerms(YAML::Node node);
  void assembleObsFrame();
  void stackObsGlobal();

  YAML::Node configNode_;
  
  std::unique_ptr<IPolicyRunner> policy_runner_;
  size_t input_dim_, output_dim_;
  std::vector<float> input_buf_, output_buf_;

  float policy_dt_ = 0.02f;
  std::vector<size_t> joint_ids_map_;
  std::vector<float> stiffness_, damping_, last_action_;

  std::unordered_map<std::string, Processor> commands_, actions_;

  struct ObsTerm {
    std::string name;
    size_t dim = 0;
    size_t offset = 0;
    std::optional<Processor> proc;   // term 自己的 process
  };
  std::vector<ObsTerm> obs_terms_;

  size_t stack_len_ = 1;
  std::string stack_order_ = "oldest_first";
  size_t obs_dim_ = 0;
  std::vector<float> obs_now_;                 // 单帧 obs（obs_dim_）
  std::deque<std::vector<float>> obs_hist_;      // 历史帧

  size_t DimOfObsTerm(const std::string& name) {
    if (name == "base_ang_vel") return 3;
    if (name == "projected_gravity") return 3;
    if (name == "velocity_commands") return 3;
    if (name == "joint_pos") return robot_model_.nJoints();
    if (name == "joint_vel") return robot_model_.nJoints();
    if (name == "last_action") return output_dim_;
    throw std::runtime_error("Unknown obs term name: " + name);
  }
};

} // namespace legged_rl_deploy
