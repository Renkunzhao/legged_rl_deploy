#pragma once
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

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

// ---------- Observations enums ----------
enum class ObsStackMethod {
  ConcatThenStack,
  StackThenConcat,
};

enum class ObsStackOrder {
  newest_first,
  oldest_first,
};

enum class ObsPreprocessOp {
  clip,
  scale,
};

inline ObsStackMethod ParseObsStackMethod(const std::string& s) {
  if (s == "ConcatThenStack") return ObsStackMethod::ConcatThenStack;
  if (s == "StackThenConcat") return ObsStackMethod::StackThenConcat;
  throw std::runtime_error("Unknown stack_method: " + s);
}

inline ObsStackOrder ParseObsStackOrder(const std::string& s) {
  if (s == "newest_first") return ObsStackOrder::newest_first;
  if (s == "oldest_first") return ObsStackOrder::oldest_first;
  throw std::runtime_error("Unknown stack_order: " + s);
}

inline ObsPreprocessOp ParseObsPreprocessOp(const std::string& s) {
  if (s == "clip") return ObsPreprocessOp::clip;
  if (s == "scale") return ObsPreprocessOp::scale;
  throw std::runtime_error("Unknown preprocess op: " + s);
}

// ---------- Observations ----------
struct ObsTerm {
  ParamMap params;               // YAML params: {}
  std::vector<double> clip;      // size 2
  std::vector<double> scale;     // dim depends on term
  int history_length = 1;
};

struct ObservationsConfig {
  // ✅ NEW: from YAML
  ObsStackMethod stack_method = ObsStackMethod::ConcatThenStack; // default
  ObsStackOrder stack_order   = ObsStackOrder::newest_first;     // default
  std::vector<ObsPreprocessOp> preprocess_order = {ObsPreprocessOp::clip, ObsPreprocessOp::scale};

  // terms: base_ang_vel / projected_gravity / ...
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
