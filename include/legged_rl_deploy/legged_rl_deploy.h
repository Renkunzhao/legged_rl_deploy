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
  void updatePolicy();

  YAML::Node configNode_;
  
  std::unique_ptr<IPolicyRunner> policy_runner_;
  size_t input_dim_, output_dim_;
  std::vector<float> input_buf_, output_buf_;

  bool clip_final_tau_ = true;
  float policy_dt_ = 0.02f;
  std::vector<size_t> joint_ids_map_;
  std::vector<float> stiffness_, damping_, last_action_;

  std::unordered_map<std::string, Processor> commands_, actions_;

  struct ObsTerm {
    std::string name;
    size_t dim = 0;
    size_t offset = 0;
    std::optional<Processor> proc;   // term 自己的 process
    YAML::Node params; 
  };
  std::vector<ObsTerm> obs_terms_;

  size_t stack_len_ = 1;
  std::string stack_order_ = "oldest_first";
  size_t obs_dim_ = 0;
  std::vector<float> obs_now_;                 // 单帧 obs（obs_dim_）
  std::deque<std::vector<float>> obs_hist_;      // 历史帧

  void calculateObsTerm(ObsTerm& term) {
    if (term.name == "constants") {
      if (!term.params || !term.params["vec"]) {
        throw std::runtime_error(
          "[LeggedRLDeploy] " + term.name + " requires params.vec");
      }
      term.dim = term.params["vec"].as<std::vector<float>>().size();
    }
    if (term.name == "joystick_buttons") {
      if (!term.params || !term.params["keys"]) {
        throw std::runtime_error(
          "[LeggedRLDeploy] " + term.name + " requires params.keys");
      }
      term.dim = term.params["keys"].as<std::vector<std::string>>().size();
    }
    if (term.name == "gait_phase_2") term.dim = 2;
    if (term.name == "base_ang_vel_W") term.dim = 3;
    if (term.name == "base_ang_vel_B") term.dim = 3;
    if (term.name == "projected_gravity") term.dim = 3;
    if (term.name == "eulerZYX_rpy") term.dim = 3;  // eulerZYX in rpy order
    if (term.name == "velocity_commands") term.dim = 3;
    if (term.name == "joint_pos") term.dim = robot_model_.nJoints();
    if (term.name == "joint_vel") term.dim = robot_model_.nJoints();
    if (term.name == "last_action") term.dim = output_dim_;
  }
};

} // namespace legged_rl_deploy
