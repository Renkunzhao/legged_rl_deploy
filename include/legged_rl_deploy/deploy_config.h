#pragma once
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace legged_rl_deploy {

// 用于保存 params: {command_name: base_velocity} 这种
using ParamMap = std::unordered_map<std::string, std::string>;

// ---------- Commands ----------
struct CommandRangesBaseVelocity {
  std::vector<double> lin_vel_x;  // size 2
  std::vector<double> lin_vel_y;  // size 2
  std::vector<double> ang_vel_z;  // size 2
  // heading: null -> optional
  std::optional<std::vector<double>> heading;  // 有些项目会是 [min,max] 或者单值
};

struct CommandsConfig {
  CommandRangesBaseVelocity base_velocity;
};

// ---------- Actions ----------
struct JointPositionActionConfig {
  // YAML: actions: JointPositionAction: ...
  // clip: list of [low, high] for each joint
  // 这里用两个 12-d 向量更好用（也更好检查维度一致性）
  std::vector<double> clip_low;   // size act_dim
  std::vector<double> clip_high;  // size act_dim

  std::vector<std::string> joint_names;   // e.g. [".*"]
  std::vector<double> scale;              // size act_dim
  std::vector<double> offset;             // size act_dim

  // joint_ids: null 允许为空
  std::optional<std::vector<int>> joint_ids;
};

// actions 未来可能不止一种类型：TorqueAction、ResidualAction 等
struct ActionsConfig {
  std::optional<JointPositionActionConfig> joint_position;
};

// ---------- Observations ----------
struct ObsTerm {
  ParamMap params;               // YAML 的 params: {}
  std::vector<double> clip;      // size 2 (low, high)
  std::vector<double> scale;     // 维度取决于 obs term
  int history_length = 1;
};

// 观测项以 name 为 key：base_ang_vel / projected_gravity / ...
struct ObservationsConfig {
  std::vector<std::pair<std::string, ObsTerm>> terms;
};

// ---------- Top-level Deploy Config ----------
struct DeployConfig {
  // Robot mapping
  std::vector<int> joint_ids_map;              // size 12

  // Control / timing
  double step_dt = 0.02;                       // e.g. 50 Hz
  std::vector<double> stiffness;               // size 12
  std::vector<double> damping;                 // size 12
  std::vector<double> default_joint_pos;       // size 12

  // Submodules (obs/action 独立)
  CommandsConfig commands;
  ActionsConfig actions;
  ObservationsConfig observations;
};

}  // namespace legged_rl_deploy
