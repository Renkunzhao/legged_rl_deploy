#pragma once

#include <string>
#include <yaml-cpp/yaml.h>

#include "legged_rl_deploy/policy/i_policy_runner.h"
#include "legged_rl_deploy/deploy_config.h"
#include "legged_rl_deploy/observation_registry.h"
#include "legged_rl_deploy/observation_assembler.h"
#include "legged_rl_deploy/action_processor.h"

#include <unitree_lowlevel/lowlevel_controller.h>
#include <legged_model/LeggedModel.h>

namespace legged_rl_deploy {

class LeggedRLDeploy : public LowLevelController {
public:
  LeggedRLDeploy(std::string configFile);

private:
//   void updateBaseState() override;
  void initHighController() override;
//   void resetHighController() override;
  void updateHighController() override;
//   void log() override;

  void registryAllObs();

  YAML::Node configNode_;
  
  DeployConfig cfg_;
  std::unique_ptr<IPolicyRunner> policy_runner_;
  int obs_dim_, act_dim_;
  std::vector<float> obs_buf_, act_buf_;
  ObservationRegistry obs_registry_;
  std::unique_ptr<ObservationAssembler> obs_asm_;
  std::unique_ptr<JointPositionActionProcessor> act_proc_;
  std::vector<float> last_action_;
  std::vector<float> default_joint_pos_f_;
  std::vector<float> kp_robot_;
  std::vector<float> kd_robot_;

};

} // namespace legged_rl_deploy
