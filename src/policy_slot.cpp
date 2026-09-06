#include "legged_rl_deploy/policy_slot.h"

#include "legged_rl_deploy/external/ros_float32_tensor_input.h"
#include "legged_rl_deploy/external/ros_image_tensor_input.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include "legged_rl_deploy/motion/local_mimic_adapter.h"
#include "legged_rl_deploy/motion/redis_mimic_adapter.h"
#include "legged_rl_deploy/policy/policy_factory.h"
#include <legged_base/Utils.h>

namespace legged_rl_deploy {

namespace {

std::vector<float> quatToRpy(const Eigen::Quaterniond& q_in) {
  const double x = q_in.x();
  const double y = q_in.y();
  const double z = q_in.z();
  const double w = q_in.w();

  const double t0 = 2.0 * (w * x + y * z);
  const double t1 = 1.0 - 2.0 * (x * x + y * y);
  const double roll = std::atan2(t0, t1);

  double t2 = 2.0 * (w * y - z * x);
  t2 = std::clamp(t2, -1.0, 1.0);
  const double pitch = std::asin(t2);

  const double t3 = 2.0 * (w * z + x * y);
  const double t4 = 1.0 - 2.0 * (y * y + z * z);
  const double yaw = std::atan2(t3, t4);

  return {static_cast<float>(roll), static_cast<float>(pitch),
          static_cast<float>(yaw)};
}

size_t maxLag(const std::vector<size_t>& lags) {
  if (lags.empty()) return 0;
  return *std::max_element(lags.begin(), lags.end());
}

} // namespace

PolicySlot::PolicySlot(const std::string& name, const YAML::Node& policyNode,
                       const LeggedModel& model, rclcpp::Node& node)
    : name_(name), policyNode_(policyNode), robot_model_(model), node_(node) {}

void PolicySlot::init() {
  const auto& pnode = policyNode_;

  const std::string backend = pnode["backend"].as<std::string>("ort");
  const std::string model_path =
      legged_base::getEnv("WORKSPACE") + "/" + pnode["model_path"].as<std::string>();

  policy_runner_ = makePolicyRunner(backend);
  policy_runner_->load(model_path, pnode);
  input_dim_ = policy_runner_->observationDim();
  output_dim_ = policy_runner_->actionDim();
  if (input_dim_ == 0) {
    throw std::runtime_error("[PolicySlot:" + name_ +
                             "] model requires an observations input");
  }
  std::cout << "[PolicySlot:" << name_ << "] Policy loaded (" << backend << ")."
            << std::endl;

  input_buf_.assign(input_dim_, 0.0f);
  output_buf_.assign(output_dim_, 0.0f);
  raw_output_.assign(output_dim_, 0.0f);

  policy_dt_ = pnode["policy_dt"].as<float>(0.02f);
  joint_ids_map_ = pnode["joint_ids_map"].as<std::vector<size_t>>();
  stiffness_ = pnode["stiffness"].as<std::vector<float>>();
  damping_ = pnode["damping"].as<std::vector<float>>();
  last_action_.assign(output_dim_, 0.0f);

  if (pnode["commands"]) {
    for (auto it = pnode["commands"].begin(); it != pnode["commands"].end(); ++it) {
      const std::string cname = it->first.as<std::string>();
      auto maybe = Processor::TryLoad(it->second);
      if (maybe) commands_.emplace(cname, std::move(*maybe));
    }
    const YAML::Node base_velocity = pnode["commands"]["base_velocity"];
    if (base_velocity) {
      if (!base_velocity.IsMap()) {
        throw std::runtime_error("[PolicySlot:" + name_ +
                                 "] commands.base_velocity must be a map");
      }
      const YAML::Node rate_limit = base_velocity["rate_limit"];
      if (rate_limit) {
        velocity_rate_limit_ = rate_limit.as<std::vector<float>>();
        if (velocity_rate_limit_.size() != 3 ||
            std::any_of(velocity_rate_limit_.begin(), velocity_rate_limit_.end(),
                        [](float value) {
                          return !std::isfinite(value) || value <= 0.0f;
                        })) {
          throw std::runtime_error(
              "[PolicySlot:" + name_ +
              "] base_velocity.rate_limit must contain three finite positive values");
        }
      }
    }
  }

  if (pnode["actions"]) {
    for (auto it = pnode["actions"].begin(); it != pnode["actions"].end(); ++it) {
      const std::string aname = it->first.as<std::string>();
      auto maybe = Processor::TryLoad(it->second);
      if (maybe) actions_.emplace(aname, std::move(*maybe));
    }
  }

  const YAML::Node observations = pnode["observations"];
  if (!observations || !observations.IsMap()) {
    throw std::runtime_error("[PolicySlot:" + name_ + "] observations missing");
  }
  if (observations["stack"]) {
    throw std::runtime_error("[PolicySlot:" + name_ +
                             "] observations.stack is no longer supported; use "
                             "term length/order/history_warmup or observations.assemble");
  }
  if (observations["layout"]) {
    throw std::runtime_error("[PolicySlot:" + name_ +
                             "] observations.layout is no longer supported; use "
                             "observations.assemble");
  }
  if (observations["history_warmup"]) {
    throw std::runtime_error("[PolicySlot:" + name_ +
                             "] observations.history_warmup moved to each term or "
                             "assemble block");
  }

  registryObsTerms(observations["terms"]);
  parseAssemble(observations);
  computeTermHistoryCapacities();
  initMimicSource();
  initExternalInputs();

  std::cout << "[PolicySlot:" << name_ << "] init done. input_dim=" << input_dim_
            << " output_dim=" << output_dim_ << std::endl;
}

void PolicySlot::reset(const LeggedState& state) {
  std::fill(input_buf_.begin(), input_buf_.end(), 0.0f);
  gait_phase_motion_active_ = false;
  gait_phase_elapsed_sec_ = 0.0f;
  std::fill(last_action_.begin(), last_action_.end(), 0.0f);
  std::fill(obs_now_.begin(), obs_now_.end(), 0.0f);
  std::fill(velocity_command_.begin(), velocity_command_.end(), 0.0f);
  for (auto& now : term_now_) std::fill(now.begin(), now.end(), 0.0f);
  for (auto& hist : term_hist_) hist.clear();
  has_valid_output_ = false;
  policy_runner_->reset();

  if (output_buf_.size() != output_dim_) {
    output_buf_.assign(output_dim_, 0.0f);
  }
  const auto& q = state.joint_pos();
  for (size_t i = 0; i < output_dim_; ++i) {
    if (i < joint_ids_map_.size()) {
      const size_t j = joint_ids_map_[i];
      if (j < static_cast<size_t>(q.size())) {
        output_buf_[i] = static_cast<float>(q[j]);
        continue;
      }
    }
    output_buf_[i] = 0.0f; 
  }

  if (mimic_source_) {
    mimic_source_->reset(state, policy_dt_);
  }

  std::cout << "[PolicySlot:" << name_ << "] reset." << std::endl;
}

void PolicySlot::registryObsTerms(YAML::Node node) {
  obs_terms_.clear();
  obs_term_indices_.clear();
  term_default_lags_.clear();
  has_mimic_term_ = false;
  mimic_params_ = YAML::Node();

  if (!node || !node.IsMap()) {
    throw std::runtime_error("[PolicySlot:" + name_ +
                             "] observations.terms missing or not a map");
  }

  size_t off = 0;
  for (auto it = node.begin(); it != node.end(); ++it) {
    ObsTerm t;
    t.name = it->first.as<std::string>();

    YAML::Node term_node = it->second;
    t.params = term_node["params"];
    if (!t.params || t.params.IsNull()) {
      t.params = YAML::Node(YAML::NodeType::Map);
    }

    calculateObsTerm(t);
    t.offset = off;
    off += t.dim;

    t.has_default_length = static_cast<bool>(term_node["length"]);
    t.default_length = term_node["length"].as<size_t>(1);
    if (t.default_length == 0) {
      throw std::runtime_error("[PolicySlot:" + name_ + "] observations.terms." + t.name +
                               ".length must be > 0");
    }
    t.default_order = parseOrderSpec(term_node["order"]);
    t.default_history_warmup =
        parseWarmup(term_node["history_warmup"], HistoryWarmupMode::RepeatFirst);

    if (t.default_order.mode == OrderMode::Explicit && t.has_default_length &&
        t.default_length != t.default_order.lags.size()) {
      throw std::runtime_error("[PolicySlot:" + name_ + "] observations.terms." + t.name +
                               ".length must match explicit order size");
    }

    auto maybe = Processor::TryLoad(term_node);
    if (maybe) t.proc = std::move(*maybe);

    obs_term_indices_.emplace(t.name, obs_terms_.size());
    if (t.name == "mimic") {
      has_mimic_term_ = true;
      mimic_params_ = t.params;
    }
    obs_terms_.push_back(std::move(t));
  }

  obs_dim_ = off;
  obs_now_.assign(obs_dim_, 0.0f);
  term_now_.assign(obs_terms_.size(), {});
  term_zero_.assign(obs_terms_.size(), {});
  term_hist_.assign(obs_terms_.size(), {});
  term_history_capacity_.assign(obs_terms_.size(), 0);
  term_default_lags_.reserve(obs_terms_.size());

  for (size_t i = 0; i < obs_terms_.size(); ++i) {
    const auto& term = obs_terms_[i];
    term_now_[i].assign(term.dim, 0.0f);
    term_zero_[i].assign(term.dim, 0.0f);
    term_default_lags_.push_back(resolveLags(term.default_order, term.default_length));
  }
}

void PolicySlot::calculateObsTerm(ObsTerm& term) {
  if (term.name == "constants") {
    if (!term.params || !term.params["vec"]) {
      throw std::runtime_error("[PolicySlot:" + name_ + "] constants requires params.vec");
    }
    term.dim = term.params["vec"].as<std::vector<float>>().size();
    return;
  }

  if (term.name == "joystick_buttons") {
    if (!term.params || !term.params["keys"]) {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] joystick_buttons requires params.keys");
    }
    term.dim = term.params["keys"].as<std::vector<std::string>>().size();
    return;
  }

  if (term.name == "mimic") {
    const bool has_terms = term.params["terms"] && term.params["terms"].IsSequence();
    const std::vector<std::string> mimic_terms = loadMimicTerms(term.params);
    size_t inferred = 0;
    for (const auto& t : mimic_terms) {
      if (t == "joint_pos" || t == "joint_vel") {
        inferred += robot_model_.nJoints();
      } else if (t == "motion_anchor_ori_b") {
        inferred += 6;
      } else {
        throw std::runtime_error("[PolicySlot:" + name_ +
                                 "] mimic.params.terms contains unsupported term: " + t);
      }
    }

    if (term.params["dim"]) {
      term.dim = term.params["dim"].as<size_t>();
      if (term.dim == 0) {
        throw std::runtime_error("[PolicySlot:" + name_ + "] mimic.params.dim must be > 0");
      }
      if (has_terms && term.dim != inferred) {
        throw std::runtime_error(
            "[PolicySlot:" + name_ + "] mimic.params.dim mismatch: inferred=" +
            std::to_string(inferred) + " cfg=" + std::to_string(term.dim));
      }
    } else {
      term.dim = inferred;
    }
    return;
  }

  if (term.name == "gait_phase_2") term.dim = 2;
  if (term.name == "hop_command") term.dim = 1;
  if (term.name == "base_pos") term.dim = 3;
  if (term.name == "base_quat_w") term.dim = 4;
  if (term.name == "base_lin_vel_W") term.dim = 3;
  if (term.name == "base_lin_vel_B") term.dim = 3;
  if (term.name == "base_ang_vel_W") term.dim = 3;
  if (term.name == "base_ang_vel_B") term.dim = 3;
  if (term.name == "projected_gravity") term.dim = 3;
  if (term.name == "eulerZYX_rpy") term.dim = 3;
  if (term.name == "roll_pitch") term.dim = 2;
  if (term.name == "velocity_commands") term.dim = 3;
  if (term.name == "velocity_support_commands") {
    if (commands_.find("support_width") == commands_.end()) {
      throw std::runtime_error(
          "[PolicySlot:" + name_ +
          "] velocity_support_commands requires commands.support_width");
    }
    term.dim = 4;
  }
  if (term.name == "joint_pos") term.dim = robot_model_.nJoints();
  if (term.name == "joint_vel") term.dim = robot_model_.nJoints();
  if (term.name == "last_action") term.dim = output_dim_;

  if (term.dim == 0) {
    throw std::runtime_error("[PolicySlot:" + name_ + "] Unknown obs term: " + term.name);
  }
}

void PolicySlot::parseAssemble(const YAML::Node& observations) {
  assemble_blocks_.clear();
  use_assemble_ = false;

  const YAML::Node assemble = observations["assemble"];
  if (!assemble) {
    size_t expected_dim = 0;
    for (size_t i = 0; i < obs_terms_.size(); ++i) {
      expected_dim += obs_terms_[i].dim * term_default_lags_[i].size();
    }
    if (expected_dim != input_dim_) {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] input_dim mismatch: default term-major expected=" +
                               std::to_string(expected_dim) + " cfg input_dim=" +
                               std::to_string(input_dim_));
    }
    return;
  }

  if (!assemble.IsSequence() || assemble.size() == 0) {
    throw std::runtime_error("[PolicySlot:" + name_ +
                             "] observations.assemble must be a non-empty sequence");
  }

  use_assemble_ = true;
  size_t expected_dim = 0;
  for (const auto& bnode : assemble) {
    if (!bnode.IsMap()) {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] each observations.assemble block must be a map");
    }

    AssembleBlock block;
    const YAML::Node terms = bnode["terms"];
    if (!terms || !terms.IsSequence() || terms.size() == 0) {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] assemble block requires non-empty terms sequence");
    }

    for (const auto& tnode : terms) {
      const std::string tname = tnode.as<std::string>();
      const auto it = obs_term_indices_.find(tname);
      if (it == obs_term_indices_.end()) {
        throw std::runtime_error("[PolicySlot:" + name_ +
                                 "] assemble block unknown term: " + tname);
      }
      block.term_indices.push_back(it->second);
      block.dim += obs_terms_[it->second].dim;
    }

    block.order = parseOrderSpec(bnode["order"]);
    block.has_length = static_cast<bool>(bnode["length"]);
    block.length = bnode["length"].as<size_t>(1);
    if (block.length == 0) {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] assemble block length must be > 0");
    }
    if (block.order.mode == OrderMode::Explicit && block.has_length &&
        block.length != block.order.lags.size()) {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] assemble block length must match explicit order size");
    }

    block.history_warmup =
        parseWarmup(bnode["history_warmup"], HistoryWarmupMode::RepeatFirst);
    block.lags = resolveLags(block.order, block.length);
    expected_dim += block.dim * block.lags.size();
    assemble_blocks_.push_back(std::move(block));
  }

  if (expected_dim != input_dim_) {
    throw std::runtime_error("[PolicySlot:" + name_ +
                             "] input_dim mismatch: assemble expected=" +
                             std::to_string(expected_dim) + " cfg input_dim=" +
                             std::to_string(input_dim_));
  }
}

void PolicySlot::computeTermHistoryCapacities() {
  term_history_capacity_.assign(obs_terms_.size(), 0);

  for (size_t i = 0; i < obs_terms_.size(); ++i) {
    term_history_capacity_[i] = maxLag(term_default_lags_[i]);
  }

  for (const auto& block : assemble_blocks_) {
    const size_t block_max_lag = maxLag(block.lags);
    for (const size_t term_idx : block.term_indices) {
      term_history_capacity_[term_idx] =
          std::max(term_history_capacity_[term_idx], block_max_lag);
    }
  }
}

void PolicySlot::initMimicSource() {
  if (!has_mimic_term_) return;

  const std::string source = mimic_params_["source"].as<std::string>("local");
  const size_t mimic_dim = getObsTermByName("mimic").dim;

  if (source == "local") {
    const YAML::Node local = mimic_params_["local"];
    if (!local || !local.IsMap()) {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] mimic.params.local is required for source=local");
    }

    LocalMimicAdapter::Config cfg;
    cfg.file = legged_base::getEnv("WORKSPACE") + "/" + local["file"].as<std::string>();
    cfg.fps = local["fps"].as<float>(50.0f);
    cfg.time_start = local["time_start"].as<float>(0.0f);
    cfg.time_end = local["time_end"].as<float>(-1.0f);
    cfg.hardware_order = local["hardware_order"].as<bool>(true);
    cfg.terms = loadMimicTerms(mimic_params_);

    mimic_source_ = std::make_unique<LocalMimicAdapter>(cfg, robot_model_, joint_ids_map_,
                                                         mimic_dim);
    std::cout << "[PolicySlot:" << name_ << "] mimic source: local" << std::endl;
    return;
  }

  if (source == "redis") {
    const YAML::Node redis = mimic_params_["redis"];
    if (!redis || !redis.IsMap()) {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] mimic.params.redis is required for source=redis");
    }

    RedisMimicAdapter::Config cfg;
    cfg.host = redis["host"].as<std::string>("127.0.0.1");
    cfg.port = redis["port"].as<int>(6379);
    cfg.db = redis["db"].as<int>(0);
    cfg.key = redis["key"].as<std::string>();
    cfg.timeout_ms = redis["timeout_ms"].as<int>(5);
    cfg.fallback = redis["fallback"].as<std::string>("hold_last");
    cfg.motion_start_trigger = redis["motion_start_trigger"].as<std::string>("");
    if (redis["init"] && redis["init"].IsSequence()) {
      cfg.init = redis["init"].as<std::vector<float>>();
    }

    mimic_source_ = std::make_unique<RedisMimicAdapter>(cfg, mimic_dim);
    std::cout << "[PolicySlot:" << name_ << "] mimic source: redis" << std::endl;
    return;
  }

  throw std::runtime_error("[PolicySlot:" + name_ +
                           "] mimic.params.source must be local or redis");
}

void PolicySlot::initExternalInputs() {
  runtime_inputs_.clear();
  external_input_buffers_.clear();
  external_inputs_.clear();

  const YAML::Node configs = policyNode_["external_inputs"];
  std::unordered_set<std::string> configured_names;
  for (const auto& input : policy_runner_->runtimeInputSpecs()) {
    if (input.source == "observations") {
      runtime_inputs_.push_back({input.source, input_buf_.data(), input_buf_.size()});
      continue;
    }
    constexpr const char* prefix = "external.";
    if (input.source.rfind(prefix, 0) != 0) {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] unsupported runtime input " + input.source);
    }
    const std::string external_name =
        input.source.substr(std::char_traits<char>::length(prefix));
    const YAML::Node config = configs ? configs[external_name] : YAML::Node();
    if (!config) {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] missing external_inputs." + external_name);
    }
    configured_names.emplace(external_name);
    auto inserted = external_input_buffers_.emplace(
        input.source, std::vector<float>(input.size, 0.0f));
    const std::string type = config["type"].as<std::string>();
    std::unique_ptr<TensorInput> external;
    if (type == "ros_image") {
      external = std::make_unique<RosImageTensorInput>(
          node_, input.source, config, input.shape);
    } else if (type == "ros_float32_multi_array") {
      external = std::make_unique<RosFloat32TensorInput>(
          node_, input.source, config, input.shape);
    } else {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] unsupported external input type " + type);
    }
    external_inputs_.emplace(input.source, std::move(external));
    runtime_inputs_.push_back(
        {input.source, inserted.first->second.data(), inserted.first->second.size()});
  }

  if (configs) {
    if (!configs.IsMap()) {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] external_inputs must be a map");
    }
    for (auto it = configs.begin(); it != configs.end(); ++it) {
      const std::string name = it->first.as<std::string>();
      if (configured_names.count(name) == 0) {
        throw std::runtime_error("[PolicySlot:" + name_ +
                                 "] unused external_inputs." + name);
      }
    }
  }
}

void PolicySlot::updateCommands(const unitree::common::Gamepad& gamepad) {
  std::vector<float> target{gamepad.ly, -gamepad.lx, -gamepad.rx};
  const auto processor = commands_.find("base_velocity");
  if (processor != commands_.end()) processor->second.process(target);

  if (velocity_rate_limit_.empty()) {
    velocity_command_ = std::move(target);
  } else {
    for (size_t i = 0; i < velocity_command_.size(); ++i) {
      const float max_delta = velocity_rate_limit_[i] * policy_dt_;
      const float delta = std::clamp(target[i] - velocity_command_[i],
                                     -max_delta, max_delta);
      velocity_command_[i] += delta;
    }
  }

  const auto support_width_processor = commands_.find("support_width");
  if (support_width_processor != commands_.end()) {
    std::vector<float> support_width{gamepad.ry};
    support_width_processor->second.process(support_width);
    support_width_command_ = support_width[0];
  }
}

void PolicySlot::assembleObsFrame(const LeggedState& state,
                                  const unitree::common::Gamepad& gamepad,
                                  size_t loop_cnt, double ll_dt) {
  std::fill(obs_now_.begin(), obs_now_.end(), 0.0f);
  updateCommands(gamepad);

  if (mimic_source_) {
    mimic_source_->step(state);
  }

  for (size_t term_idx = 0; term_idx < obs_terms_.size(); ++term_idx) {
    const auto& term = obs_terms_[term_idx];
    std::vector<float> v(term.dim, 0.0f);

    if (term.name == "constants") {
      v = term.params["vec"].as<std::vector<float>>();

    } else if (term.name == "joystick_buttons") {
      const auto keys = term.params["keys"].as<std::vector<std::string>>();
      for (size_t i = 0; i < keys.size(); ++i) {
        unitree::common::Button btn;
        if (keys[i] == "A")
          btn = gamepad.A;
        else if (keys[i] == "B")
          btn = gamepad.B;
        else if (keys[i] == "X")
          btn = gamepad.X;
        else if (keys[i] == "Y")
          btn = gamepad.Y;
        else if (keys[i] == "up")
          btn = gamepad.up;
        else if (keys[i] == "down")
          btn = gamepad.down;
        else if (keys[i] == "left")
          btn = gamepad.left;
        else if (keys[i] == "right")
          btn = gamepad.right;
        else if (keys[i] == "L1")
          btn = gamepad.L1;
        else if (keys[i] == "L2")
          btn = gamepad.L2;
        else if (keys[i] == "R1")
          btn = gamepad.R1;
        else if (keys[i] == "R2")
          btn = gamepad.R2;
        else if (keys[i] == "start")
          btn = gamepad.start;
        else if (keys[i] == "select")
          btn = gamepad.select;
        else
          throw std::runtime_error("[PolicySlot:" + name_ +
                                   "] Unknown joystick button: " + keys[i]);
        v[i] = btn.pressed ? 1.0f : 0.0f;
      }

    } else if (term.name == "gait_phase_2") {
      const float cycle_time = term.params["cycle_time"].as<float>();
      const float command_threshold =
          term.params["command_threshold"].as<float>(-1.0f);
      const bool reset_on_motion =
          term.params["reset_on_motion"].as<bool>(false);

      if (!std::isfinite(cycle_time) || cycle_time <= 0.0f) {
        throw std::runtime_error(
            "[PolicySlot:" + name_ +
            "] gait_phase_2 cycle_time must be finite and positive");
      }

      const float command_norm = std::sqrt(
          velocity_command_[0] * velocity_command_[0] +
          velocity_command_[1] * velocity_command_[1] +
          velocity_command_[2] * velocity_command_[2]);

      const bool is_moving =
          command_threshold < 0.0f || command_norm >= command_threshold;

      constexpr float kTwoPi = 6.28318530718f;

      if (!is_moving) {
        // Training contract: stationary phase observation is [0, 0].
        v[0] = 0.0f;
        v[1] = 0.0f;

        if (reset_on_motion) {
          gait_phase_motion_active_ = false;
          gait_phase_elapsed_sec_ = 0.0f;
        }
      } else if (reset_on_motion) {
        if (!gait_phase_motion_active_) {
          // First moving frame must be phase=0 -> [sin(0), cos(0)] = [0, 1].
          gait_phase_motion_active_ = true;
          gait_phase_elapsed_sec_ = 0.0f;

          std::cout << "[PolicySlot:" << name_
                    << "] gait phase reset on motion start: [0, 1]"
                    << std::endl;
        }

        const float phase = gait_phase_elapsed_sec_ / cycle_time;
        v[0] = std::sin(kTwoPi * phase);
        v[1] = std::cos(kTwoPi * phase);

        // assembleObsFrame() runs once per policy inference.
        gait_phase_elapsed_sec_ += policy_dt_;
      } else {
        // Preserve legacy global-phase behavior for other policies.
        const float phase =
            static_cast<float>(loop_cnt * ll_dt / cycle_time);
        v[0] = std::sin(kTwoPi * phase);
        v[1] = std::cos(kTwoPi * phase);
      }

    } else if (term.name == "hop_command") {
      v[0] = term.params["peak_height"].as<float>(0.7f);

    } else if (term.name == "base_pos") {
      v[0] = state.base_pos()[0];
      v[1] = state.base_pos()[1];
      v[2] = state.base_pos()[2];

    } else if (term.name == "base_quat_w") {
      const auto& q = state.base_quat();
      const double sign = q.w() < 0.0 ? -1.0 : 1.0;
      v[0] = static_cast<float>(sign * q.w());
      v[1] = static_cast<float>(sign * q.x());
      v[2] = static_cast<float>(sign * q.y());
      v[3] = static_cast<float>(sign * q.z());

    } else if (term.name == "base_lin_vel_W") {
      v[0] = state.base_lin_vel_W()[0];
      v[1] = state.base_lin_vel_W()[1];
      v[2] = state.base_lin_vel_W()[2];

    } else if (term.name == "base_lin_vel_B") {
      v[0] = state.base_lin_vel_B()[0];
      v[1] = state.base_lin_vel_B()[1];
      v[2] = state.base_lin_vel_B()[2];

    } else if (term.name == "base_ang_vel_W") {
      v[0] = state.base_ang_vel_W()[0];
      v[1] = state.base_ang_vel_W()[1];
      v[2] = state.base_ang_vel_W()[2];

    } else if (term.name == "base_ang_vel_B") {
      v[0] = state.base_ang_vel_B()[0];
      v[1] = state.base_ang_vel_B()[1];
      v[2] = state.base_ang_vel_B()[2];

    } else if (term.name == "eulerZYX_rpy") {
      v = quatToRpy(state.base_quat());

    } else if (term.name == "roll_pitch") {
      const std::vector<float> rpy = quatToRpy(state.base_quat());
      v[0] = rpy[0];
      v[1] = rpy[1];

    } else if (term.name == "projected_gravity") {
      Eigen::Vector3d g =
          state.base_quat().conjugate() * Eigen::Vector3d(0, 0, -1);
      v[0] = static_cast<float>(g[0]);
      v[1] = static_cast<float>(g[1]);
      v[2] = static_cast<float>(g[2]);

    } else if (term.name == "velocity_commands") {
      v = velocity_command_;

    } else if (term.name == "velocity_support_commands") {
      v[0] = velocity_command_[0];
      v[1] = velocity_command_[1];
      v[2] = velocity_command_[2];
      v[3] = support_width_command_;

    } else if (term.name == "joint_pos") {
      for (size_t i = 0; i < robot_model_.nJoints(); ++i) {
        v[i] = state.joint_pos()[joint_ids_map_[i]];
      }

    } else if (term.name == "joint_vel") {
      for (size_t i = 0; i < robot_model_.nJoints(); ++i) {
        v[i] = state.joint_vel()[joint_ids_map_[i]];
      }

    } else if (term.name == "last_action") {
      v = last_action_;

    } else if (term.name == "mimic") {
      if (!mimic_source_) {
        throw std::runtime_error("[PolicySlot:" + name_ +
                                 "] mimic term requires a configured mimic source");
      }
      mimic_source_->read(v);
      if (v.size() != term.dim) {
        throw std::runtime_error("[PolicySlot:" + name_ +
                                 "] mimic read dim mismatch");
      }

    } else {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] Unknown obs term: " + term.name);
    }

    if (term.proc) term.proc->process(v);
    term_now_[term_idx] = v;
    std::copy(v.begin(), v.end(), obs_now_.begin() + term.offset);
  }
}

void PolicySlot::assembleDefaultTermMajor() {
  size_t out = 0;
  for (size_t term_idx = 0; term_idx < obs_terms_.size(); ++term_idx) {
    for (const size_t lag : term_default_lags_[term_idx]) {
      const auto& sample =
          sampleTermAtLag(term_idx, lag, obs_terms_[term_idx].default_history_warmup);
      std::copy(sample.begin(), sample.end(), input_buf_.begin() + out);
      out += sample.size();
    }
  }

  if (out != input_buf_.size()) {
    throw std::runtime_error("[PolicySlot:" + name_ +
                             "] internal error: default term-major packed dim mismatch");
  }
}

void PolicySlot::assembleFromBlocks() {
  size_t out = 0;
  for (const auto& block : assemble_blocks_) {
    for (const size_t lag : block.lags) {
      for (const size_t term_idx : block.term_indices) {
        const auto& sample = sampleTermAtLag(term_idx, lag, block.history_warmup);
        std::copy(sample.begin(), sample.end(), input_buf_.begin() + out);
        out += sample.size();
      }
    }
  }

  if (out != input_buf_.size()) {
    throw std::runtime_error("[PolicySlot:" + name_ +
                             "] internal error: assemble packed dim mismatch");
  }
}

void PolicySlot::pushTermHistory() {
  for (size_t term_idx = 0; term_idx < obs_terms_.size(); ++term_idx) {
    const size_t cap = term_history_capacity_[term_idx];
    if (cap == 0) continue;
    auto& hist = term_hist_[term_idx];
    hist.push_back(term_now_[term_idx]);
    while (hist.size() > cap) hist.pop_front();
  }
}

void PolicySlot::updatePolicy(const LeggedState& state,
                              const unitree::common::Gamepad& gamepad,
                              size_t loop_cnt, double ll_dt) {
  assembleObsFrame(state, gamepad, loop_cnt, ll_dt);

  if (use_assemble_) {
    assembleFromBlocks();
  } else {
    assembleDefaultTermMajor();
  }
  pushTermHistory();

  for (auto& external : external_inputs_) {
    external.second->read(external_input_buffers_.at(external.first));
  }
  for (auto& input : runtime_inputs_) {
    if (input.source == "observations") {
      input.data = input_buf_.data();
    } else {
      const auto& buffer = external_input_buffers_.at(input.source);
      input.data = buffer.data();
      input.size = buffer.size();
    }
  }
  policy_runner_->infer(runtime_inputs_, raw_output_.data());

  last_action_ = raw_output_;
  output_buf_ = raw_output_;
  has_valid_output_ = true;

  auto it = actions_.find("JointPositionAction");
  if (it != actions_.end()) it->second.process(output_buf_);
}

void PolicySlot::update(const LeggedState& state,
                        const unitree::common::Gamepad& gamepad,
                        size_t loop_cnt, double ll_dt) {
  if (mimic_source_) {
    mimic_source_->onGamepad(gamepad);
  }

  const int decim = std::max(1, static_cast<int>(std::lround(policy_dt_ / ll_dt)));
  if ((loop_cnt % decim) == 0) {
    updatePolicy(state, gamepad, loop_cnt, ll_dt);
  }
}

const PolicySlot::ObsTerm& PolicySlot::getObsTermByName(
    const std::string& name) const {
  const auto it = obs_term_indices_.find(name);
  if (it == obs_term_indices_.end()) {
    throw std::runtime_error("[PolicySlot:" + name_ + "] obs term not found: " + name);
  }
  return obs_terms_[it->second];
}

PolicySlot::ObsOrderSpec PolicySlot::parseOrderSpec(const YAML::Node& node) {
  ObsOrderSpec spec;
  if (!node || node.IsNull()) return spec;

  if (node.IsScalar()) {
    const std::string order = node.as<std::string>();
    if (order == "oldest_first") {
      spec.mode = OrderMode::OldestFirst;
      return spec;
    }
    if (order == "newest_first") {
      spec.mode = OrderMode::NewestFirst;
      return spec;
    }
    throw std::runtime_error("order must be oldest_first, newest_first, or an integer "
                             "sequence");
  }

  if (!node.IsSequence() || node.size() == 0) {
    throw std::runtime_error("explicit order must be a non-empty integer sequence");
  }

  spec.mode = OrderMode::Explicit;
  spec.lags.reserve(node.size());
  for (const auto& lnode : node) {
    const int lag = lnode.as<int>();
    if (lag < 0) {
      throw std::runtime_error("explicit order entries must be >= 0");
    }
    spec.lags.push_back(static_cast<size_t>(lag));
  }
  return spec;
}

std::vector<size_t> PolicySlot::resolveLags(const ObsOrderSpec& spec, size_t length) {
  if (spec.mode == OrderMode::Explicit) return spec.lags;
  if (length == 0) {
    throw std::runtime_error("length must be > 0 when order is not explicit");
  }

  std::vector<size_t> lags;
  lags.reserve(length);
  if (spec.mode == OrderMode::NewestFirst) {
    for (size_t lag = 0; lag < length; ++lag) lags.push_back(lag);
    return lags;
  }

  for (size_t lag = length; lag-- > 0;) lags.push_back(lag);
  return lags;
}

PolicySlot::HistoryWarmupMode PolicySlot::parseWarmup(
    const YAML::Node& node, HistoryWarmupMode default_mode) {
  if (!node || node.IsNull()) return default_mode;

  const std::string mode = node.as<std::string>();
  if (mode == "repeat_first") return HistoryWarmupMode::RepeatFirst;
  if (mode == "zero") return HistoryWarmupMode::Zero;
  throw std::runtime_error("history_warmup must be repeat_first or zero");
}

const std::vector<float>& PolicySlot::sampleTermAtLag(
    size_t term_idx, size_t lag, HistoryWarmupMode warmup) const {
  if (lag == 0) return term_now_[term_idx];

  const auto& hist = term_hist_[term_idx];
  if (hist.size() >= lag) {
    return hist[hist.size() - lag];
  }
  if (warmup == HistoryWarmupMode::Zero) {
    return term_zero_[term_idx];
  }
  if (!hist.empty()) {
    return hist.front();
  }
  return term_now_[term_idx];
}

std::vector<std::string> PolicySlot::loadMimicTerms(const YAML::Node& params) {
  if (params["terms"] && params["terms"].IsSequence()) {
    return params["terms"].as<std::vector<std::string>>();
  }
  return {"joint_pos", "joint_vel", "motion_anchor_ori_b"};
}

} // namespace legged_rl_deploy
