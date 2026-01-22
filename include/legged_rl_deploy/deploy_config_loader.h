#pragma once

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "legged_rl_deploy/deploy_config.h"

namespace legged_rl_deploy {

inline std::string JoinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  return a + "." + b;
}

struct YamlLoadError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline void Require(bool cond, const std::string& msg) {
  if (!cond) throw YamlLoadError(msg);
}

inline YAML::Node RequireNode(const YAML::Node& parent,
                              const std::string& key,
                              const std::string& path) {
  Require(parent.IsDefined() && !parent.IsNull() && parent.IsMap(),
          "Expected a map node at '" + path + "', but got null/non-map.");
  YAML::Node n = parent[key];
  Require(n.IsDefined(), "Missing required key '" + key + "' at '" + path + "'.");
  return n;
}

template <typename T>
inline T AsScalar(const YAML::Node& n, const std::string& path) {
  Require(n.IsDefined() && !n.IsNull() && n.IsScalar(),
          "Expected scalar at '" + path + "'.");
  try {
    return n.as<T>();
  } catch (const std::exception& e) {
    throw YamlLoadError("Failed to parse scalar at '" + path + "': " + e.what());
  }
}

template <typename T>
inline std::vector<T> AsVector(const YAML::Node& n, const std::string& path) {
  Require(n.IsDefined() && !n.IsNull() && n.IsSequence(),
          "Expected sequence at '" + path + "'.");
  std::vector<T> out;
  out.reserve(n.size());
  for (size_t i = 0; i < n.size(); ++i) {
    const auto& item = n[i];
    try {
      out.push_back(item.as<T>());
    } catch (const std::exception& e) {
      throw YamlLoadError("Failed to parse element [" + std::to_string(i) +
                          "] at '" + path + "': " + e.what());
    }
  }
  return out;
}

template <typename T>
inline std::vector<T> AsVectorFixed(const YAML::Node& n, size_t expected_size,
                                   const std::string& path) {
  auto v = AsVector<T>(n, path);
  Require(v.size() == expected_size,
          "Size mismatch at '" + path + "': expected " +
              std::to_string(expected_size) + ", got " +
              std::to_string(v.size()) + ".");
  return v;
}

inline std::optional<std::vector<double>> AsOptionalVectorDouble(const YAML::Node& n,
                                                                 const std::string& path) {
  if (!n.IsDefined() || n.IsNull()) return std::nullopt;
  return AsVector<double>(n, path);
}

inline ParamMap AsParamMapStringString(const YAML::Node& n, const std::string& path) {
  ParamMap out;
  if (!n.IsDefined() || n.IsNull()) return out;  // 允许缺省 -> {}
  Require(n.IsMap(), "Expected map at '" + path + "'.");
  for (auto it = n.begin(); it != n.end(); ++it) {
    const auto k = it->first;
    const auto v = it->second;
    Require(k.IsDefined() && !k.IsNull() && k.IsScalar(),
            "Expected scalar key in params at '" + path + "'.");
    const std::string key = k.as<std::string>();

    std::string val_str;
    if (!v.IsDefined() || v.IsNull()) {
      val_str = "null";
    } else if (v.IsScalar()) {
      val_str = v.as<std::string>();
    } else {
      val_str = YAML::Dump(v);
    }
    out[key] = val_str;
  }
  return out;
}

// --------- Sub-loaders ---------

inline CommandRangesBaseVelocity LoadCommandRangesBaseVelocity(const YAML::Node& ranges,
                                                               const std::string& path) {
  Require(ranges.IsDefined() && !ranges.IsNull() && ranges.IsMap(),
          "Expected map at '" + path + "'.");
  CommandRangesBaseVelocity r;
  r.lin_vel_x = AsVectorFixed<double>(RequireNode(ranges, "lin_vel_x", path), 2,
                                      JoinPath(path, "lin_vel_x"));
  r.lin_vel_y = AsVectorFixed<double>(RequireNode(ranges, "lin_vel_y", path), 2,
                                      JoinPath(path, "lin_vel_y"));
  r.ang_vel_z = AsVectorFixed<double>(RequireNode(ranges, "ang_vel_z", path), 2,
                                      JoinPath(path, "ang_vel_z"));

  YAML::Node heading = ranges["heading"];
  if (!heading.IsDefined() || heading.IsNull()) {
    r.heading = std::nullopt;
  } else {
    if (heading.IsSequence()) {
      r.heading = AsVector<double>(heading, JoinPath(path, "heading"));
    } else if (heading.IsScalar()) {
      r.heading = std::vector<double>{heading.as<double>()};
    } else {
      throw YamlLoadError("Unsupported type for 'heading' at '" +
                          JoinPath(path, "heading") + "'.");
    }
  }
  return r;
}

inline CommandsConfig LoadCommandsConfig(const YAML::Node& commands, const std::string& path) {
  Require(commands.IsDefined() && !commands.IsNull() && commands.IsMap(),
          "Expected map at '" + path + "'.");
  CommandsConfig cfg;

  YAML::Node base_vel = RequireNode(commands, "base_velocity", path);
  YAML::Node ranges   = RequireNode(base_vel, "ranges", JoinPath(path, "base_velocity"));
  cfg.base_velocity = LoadCommandRangesBaseVelocity(ranges, JoinPath(path, "base_velocity.ranges"));
  return cfg;
}

inline JointPositionActionConfig LoadJointPositionAction(const YAML::Node& jpa,
                                                         size_t act_dim,
                                                         const std::string& path) {
  Require(jpa.IsDefined() && !jpa.IsNull() && jpa.IsMap(),
          "Expected map at '" + path + "'.");
  JointPositionActionConfig cfg;

  YAML::Node clip = RequireNode(jpa, "clip", path);
  Require(clip.IsSequence(), "Expected sequence at '" + JoinPath(path, "clip") + "'.");
  Require(clip.size() == act_dim,
          "Size mismatch at '" + JoinPath(path, "clip") + "': expected " +
              std::to_string(act_dim) + ", got " + std::to_string(clip.size()) + ".");

  cfg.clip_low.resize(act_dim);
  cfg.clip_high.resize(act_dim);
  for (size_t i = 0; i < act_dim; ++i) {
    YAML::Node pair = clip[i];
    Require(pair.IsDefined() && !pair.IsNull() && pair.IsSequence() && pair.size() == 2,
            "Expected [low, high] at '" + JoinPath(path, "clip[" + std::to_string(i) + "]") + "'.");
    cfg.clip_low[i] = pair[0].as<double>();
    cfg.clip_high[i] = pair[1].as<double>();
  }

  cfg.joint_names = AsVector<std::string>(RequireNode(jpa, "joint_names", path),
                                         JoinPath(path, "joint_names"));
  cfg.scale  = AsVectorFixed<double>(RequireNode(jpa, "scale", path),  act_dim,
                                     JoinPath(path, "scale"));
  cfg.offset = AsVectorFixed<double>(RequireNode(jpa, "offset", path), act_dim,
                                     JoinPath(path, "offset"));

  YAML::Node joint_ids = jpa["joint_ids"];
  if (!joint_ids.IsDefined() || joint_ids.IsNull()) {
    cfg.joint_ids = std::nullopt;
  } else {
    cfg.joint_ids = AsVector<int>(joint_ids, JoinPath(path, "joint_ids"));
  }

  return cfg;
}

inline ActionsConfig LoadActionsConfig(const YAML::Node& actions,
                                       size_t act_dim,
                                       const std::string& path) {
  ActionsConfig cfg;
  if (!actions.IsDefined() || actions.IsNull()) return cfg;
  Require(actions.IsMap(), "Expected map at '" + path + "'.");

  YAML::Node jpa = actions["JointPositionAction"];
  if (jpa.IsDefined() && !jpa.IsNull()) {
    cfg.joint_position = LoadJointPositionAction(jpa, act_dim, JoinPath(path, "JointPositionAction"));
  } else {
    cfg.joint_position = std::nullopt;
  }
  return cfg;
}

inline ObsTerm LoadObsTerm(const YAML::Node& term, const std::string& path) {
  Require(term.IsDefined() && !term.IsNull() && term.IsMap(),
          "Expected map at '" + path + "'.");
  ObsTerm t;

  t.params = AsParamMapStringString(term["params"], JoinPath(path, "params"));
  t.clip   = AsVectorFixed<double>(RequireNode(term, "clip", path), 2, JoinPath(path, "clip"));
  t.scale  = AsVector<double>(RequireNode(term, "scale", path), JoinPath(path, "scale"));

  YAML::Node hl = term["history_length"];
  if (hl.IsDefined() && !hl.IsNull()) {
    t.history_length = AsScalar<int>(hl, JoinPath(path, "history_length"));
    Require(t.history_length >= 1,
            "history_length must be >= 1 at '" + JoinPath(path, "history_length") + "'.");
  }
  return t;
}

inline ObservationsConfig LoadObservationsConfig(const YAML::Node& obs, const std::string& path) {
  ObservationsConfig cfg;
  if (!obs.IsDefined() || obs.IsNull()) return cfg;
  Require(obs.IsMap(), "Expected map at '" + path + "'.");

  cfg.terms.clear();
  cfg.terms.reserve(obs.size());

  // duplicate check
  std::unordered_map<std::string, int> seen;

  for (auto it = obs.begin(); it != obs.end(); ++it) {
    Require(it->first.IsScalar(), "Observation term name must be scalar at '" + path + "'.");
    const std::string name = it->first.as<std::string>();

    if (++seen[name] > 1) {
      throw YamlLoadError("Duplicate observation term name '" + name + "' at '" + path + "'.");
    }

    ObsTerm term = LoadObsTerm(it->second, path + "." + name);
    cfg.terms.emplace_back(name, std::move(term));
  }

  return cfg;
}

// --------- Top-level ---------

inline DeployConfig LoadDeployConfigFromNode(const YAML::Node& root,
                                            size_t act_dim = 12) {
  Require(root.IsDefined() && !root.IsNull() && root.IsMap(), "Root YAML must be a map.");

  DeployConfig cfg;

  cfg.joint_ids_map = AsVectorFixed<int>(RequireNode(root, "joint_ids_map", "<root>"),
                                        act_dim, "joint_ids_map");
  cfg.step_dt = AsScalar<double>(RequireNode(root, "step_dt", "<root>"), "step_dt");
  cfg.stiffness = AsVectorFixed<double>(RequireNode(root, "stiffness", "<root>"),
                                       act_dim, "stiffness");
  cfg.damping = AsVectorFixed<double>(RequireNode(root, "damping", "<root>"),
                                     act_dim, "damping");
  cfg.default_joint_pos = AsVectorFixed<double>(RequireNode(root, "default_joint_pos", "<root>"),
                                               act_dim, "default_joint_pos");

  cfg.commands = LoadCommandsConfig(RequireNode(root, "commands", "<root>"), "commands");
  cfg.actions  = LoadActionsConfig(root["actions"], act_dim, "actions");
  cfg.observations = LoadObservationsConfig(root["observations"], "observations");

  return cfg;
}

inline DeployConfig LoadDeployConfigFromFile(const std::string& yaml_path,
                                            size_t act_dim = 12) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(yaml_path);
  } catch (const std::exception& e) {
    throw YamlLoadError("Failed to load YAML file '" + yaml_path + "': " + e.what());
  }
  return LoadDeployConfigFromNode(root, act_dim);
}

}  // namespace legged_rl_deploy
