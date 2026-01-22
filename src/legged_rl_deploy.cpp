#include "legged_rl_deploy/legged_rl_deploy.h"

#include <algorithm>
#include <iostream>
#include <memory>

#include "legged_rl_deploy/observation_assembler.h"
#include "legged_rl_deploy/observation_registry.h"
#include "legged_rl_deploy/policy/policy_factory.h"
#include "legged_rl_deploy/deploy_config_loader.h"
#include "legged_model/Utils.h"

namespace legged_rl_deploy {

LeggedRLDeploy::LeggedRLDeploy(std::string configFile) {
  configNode_ = YAML::LoadFile(configFile);
  std::cout << "[LeggedRLDeploy] Load config from " << configFile << std::endl;
}

void LeggedRLDeploy::initHighController() {
  const auto& pnode = configNode_["policy"];
  const std::string backend   = pnode["backend"].as<std::string>("torch");
  const std::string model_path= pnode["model_path"].as<std::string>();
  obs_dim_ = pnode["obs_dim"].as<int>();
  act_dim_ = pnode["act_dim"].as<int>();

  cfg_ = LoadDeployConfigFromNode(pnode, act_dim_);

  policy_runner_ = makePolicyRunner(backend);
  policy_runner_->load(model_path, obs_dim_, act_dim_);
  std::cout << "[LeggedRLDeploy] Policy load successfully." << std::endl;

  // 3) Init ObservationAssembler (包含维度校验)
  obs_asm_ = std::make_unique<ObservationAssembler>(cfg_.observations, obs_dim_);
  obs_buf_.assign(static_cast<size_t>(obs_dim_), 0.0f);

  // 3.1) Init ObservationRegistry
  registryAllObs();

  // 4) Init ActionProcessor（JointPositionAction）
  if (!cfg_.actions.joint_position.has_value()) {
    throw std::runtime_error("actions.joint_position is missing in YAML.");
  }
  act_proc_ = std::make_unique<JointPositionActionProcessor>(
      cfg_.actions.joint_position.value(),
      cfg_.joint_ids_map,
      act_dim_);

  act_buf_.assign(static_cast<size_t>(act_dim_), 0.0f);
  last_action_.assign(static_cast<size_t>(act_dim_), 0.0f);
  default_joint_pos_f_ = LeggedAI::stdVecDoubleToFloat(cfg_.default_joint_pos);

  const int robot_dof = static_cast<int>(cfg_.joint_ids_map.size());
  kp_robot_.assign(static_cast<size_t>(robot_dof), 0.0f);
  kd_robot_.assign(static_cast<size_t>(robot_dof), 0.0f);
  for (int j_internal = 0; j_internal < robot_dof; ++j_internal) {
    const int j_robot = cfg_.joint_ids_map[static_cast<size_t>(j_internal)];
    kp_robot_[static_cast<size_t>(j_robot)] =
        static_cast<float>(cfg_.stiffness[static_cast<size_t>(j_internal)]);
    kd_robot_[static_cast<size_t>(j_robot)] =
        static_cast<float>(cfg_.damping[static_cast<size_t>(j_internal)]);
  }

  // 6) reset history buffers
  obs_asm_->Reset();

  std::cout << "[LeggedRLDeploy] init done. obs_dim=" << obs_dim_
            << " act_dim=" << act_dim_ << std::endl;
}

void LeggedRLDeploy::registryAllObs() {
  obs_registry_.Register("base_ang_vel", [this](const ObsTerm&) {
    return LeggedAI::eigenToStdVec(real_state_.base_ang_vel_B());
  });

  obs_registry_.Register("projected_gravity", [this](const ObsTerm&) {
    const Eigen::Vector3d gravity_w(0.0, 0.0, -1.0);
    const Eigen::Vector3d gravity_b = real_state_.base_quat().conjugate() * gravity_w;
    return LeggedAI::eigenToStdVec(gravity_b);
  });

  obs_registry_.Register("velocity_commands", [this](const ObsTerm& term) {
    auto it = term.params.find("command_name");
    const std::string command_name = (it == term.params.end()) ? "base_velocity" : it->second;
    if (command_name != "base_velocity") {
      throw std::runtime_error("Unsupported command_name in velocity_commands: " + command_name);
    }

    const auto& ranges = cfg_.commands.base_velocity;

    const double vx = std::clamp(static_cast<double>(gamepad_.ly),
                                 ranges.lin_vel_x[0], ranges.lin_vel_x[1]);
    const double vy = std::clamp(static_cast<double>(-gamepad_.lx),
                                 ranges.lin_vel_y[0], ranges.lin_vel_y[1]);
    const double wz = std::clamp(static_cast<double>(-gamepad_.rx),
                                 ranges.ang_vel_z[0], ranges.ang_vel_z[1]);
    return std::vector<double>{vx, vy, wz};
  });

  obs_registry_.Register("joint_pos_rel", [this](const ObsTerm&) {
    const auto q = real_state_.joint_pos();
    if (q.size() != static_cast<int>(cfg_.default_joint_pos.size())) {
      throw std::runtime_error("joint_pos_rel: joint_pos size mismatch with default_joint_pos.");
    }
    std::vector<double> out(static_cast<size_t>(q.size()));
    for (int i = 0; i < q.size(); ++i) {
      out[static_cast<size_t>(i)] =
          q[i] - cfg_.default_joint_pos[static_cast<size_t>(i)];
    }
    return out;
  });

  obs_registry_.Register("joint_vel_rel", [this](const ObsTerm&) {
    return LeggedAI::eigenToStdVec(real_state_.joint_vel());
  });
}

void LeggedRLDeploy::updateHighController() {
  // 1) Fill obs terms via registry (order by YAML)
  obs_registry_.Fill(cfg_.observations, *obs_asm_, last_action_);

  // 2) Assemble flat obs
  obs_buf_ = obs_asm_->Assemble(/*require_all_terms=*/true);

  // 3) Policy inference: obs -> act
  policy_runner_->infer(obs_buf_.data(), act_buf_.data());

  // 4) 保存 last_action（供下一 tick obs）
  last_action_ = act_buf_;

  // 5) Action post-process: policy action -> joint target (robot order)
  // （建议用 default_joint_pos_f_ 填充未控制关节）
  std::vector<float> q_des_robot =
      act_proc_->ComputeJointTargetsRobotOrder(act_buf_, default_joint_pos_f_);

  // 6) 写 lowcmd（位置控制 + PD）
  for (size_t j_robot = 0; j_robot < kp_robot_.size(); ++j_robot) {
    lowcmd_msg_.motor_cmd[j_robot].q = q_des_robot[j_robot];
    lowcmd_msg_.motor_cmd[j_robot].kp = kp_robot_[j_robot];
    lowcmd_msg_.motor_cmd[j_robot].kd = kd_robot_[j_robot];
    lowcmd_msg_.motor_cmd[j_robot].dq = 0.0;
    lowcmd_msg_.motor_cmd[j_robot].tau = 0.0;
  }
}

} // namespace legged_rl_deploy
