#include "legged_rl_deploy/policy_slot.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

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

} // namespace

PolicySlot::PolicySlot(const std::string& name, const YAML::Node& policyNode,
                       const LeggedModel& model)
    : name_(name), policyNode_(policyNode), robot_model_(model) {}

void PolicySlot::init() {
  const auto& pnode = policyNode_;

  const std::string backend = pnode["backend"].as<std::string>("ort");
  const std::string model_path =
      legged_base::getEnv("WORKSPACE") + "/" + pnode["model_path"].as<std::string>();

  input_dim_ = pnode["input_dim"].as<size_t>();
  output_dim_ = pnode["output_dim"].as<size_t>();

  policy_runner_ = makePolicyRunner(backend);
  policy_runner_->load(model_path, input_dim_, output_dim_);
  std::cout << "[PolicySlot:" << name_ << "] Policy loaded (" << backend << ")."
            << std::endl;

  input_buf_.assign(input_dim_, 0.0f);
  output_buf_.assign(output_dim_, 0.0f);

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
    stack_len_ = observations["stack"]["length"].as<size_t>(1);
    stack_order_ = observations["stack"]["order"].as<std::string>("oldest_first");
  }
  history_warmup_ = observations["history_warmup"].as<std::string>("repeat_first");

  registryObsTerms(observations["terms"]);
  parseObsLayout(observations);
  initMimicSource();

  std::cout << "[PolicySlot:" << name_ << "] init done. input_dim=" << input_dim_
            << " output_dim=" << output_dim_ << std::endl;
}

void PolicySlot::reset(const LeggedState& state) {
  std::fill(input_buf_.begin(), input_buf_.end(), 0.0f);
  std::fill(output_buf_.begin(), output_buf_.end(), 0.0f);
  std::fill(last_action_.begin(), last_action_.end(), 0.0f);
  obs_hist_.clear();

  if (mimic_source_) {
    mimic_source_->reset(state, policy_dt_);
  }

  std::cout << "[PolicySlot:" << name_ << "] reset." << std::endl;
}

void PolicySlot::registryObsTerms(YAML::Node node) {
  obs_terms_.clear();
  obs_term_indices_.clear();
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
  zeros_frame_.assign(obs_dim_, 0.0f);
  obs_hist_.clear();
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
  if (term.name == "base_ang_vel_W") term.dim = 3;
  if (term.name == "base_ang_vel_B") term.dim = 3;
  if (term.name == "projected_gravity") term.dim = 3;
  if (term.name == "eulerZYX_rpy") term.dim = 3;
  if (term.name == "roll_pitch") term.dim = 2;
  if (term.name == "velocity_commands") term.dim = 3;
  if (term.name == "joint_pos") term.dim = robot_model_.nJoints();
  if (term.name == "joint_vel") term.dim = robot_model_.nJoints();
  if (term.name == "last_action") term.dim = output_dim_;

  if (term.dim == 0) {
    throw std::runtime_error("[PolicySlot:" + name_ + "] Unknown obs term: " + term.name);
  }
}

void PolicySlot::parseObsLayout(const YAML::Node& observations) {
  if (history_warmup_ != "repeat_first" && history_warmup_ != "zero") {
    throw std::runtime_error("[PolicySlot:" + name_ +
                             "] observations.history_warmup must be repeat_first or zero");
  }

  const YAML::Node layout = observations["layout"];
  use_layout_ = layout && layout.IsMap();
  layout_blocks_.clear();
  layout_input_dim_ = 0;
  history_capacity_ = 0;

  if (!use_layout_) {
    layout_input_dim_ = obs_dim_ * stack_len_;
    if (layout_input_dim_ != input_dim_) {
      throw std::runtime_error(
          "[PolicySlot:" + name_ + "] input_dim mismatch: obs_dim=" +
          std::to_string(obs_dim_) + " stack_len=" + std::to_string(stack_len_) +
          " expected=" + std::to_string(layout_input_dim_) +
          " cfg input_dim=" + std::to_string(input_dim_));
    }
    history_capacity_ = stack_len_;
    return;
  }

  const YAML::Node blocks = layout["blocks"];
  if (!blocks || !blocks.IsSequence() || blocks.size() == 0) {
    throw std::runtime_error("[PolicySlot:" + name_ +
                             "] observations.layout.blocks must be non-empty sequence");
  }

  for (auto bnode : blocks) {
    ObsLayoutBlock block;
    const std::string kind = bnode["kind"].as<std::string>();

    if (kind == "current_frame") {
      block.kind = LayoutKind::CurrentFrame;
      block.dim = obs_dim_;
    } else if (kind == "history_frame") {
      block.kind = LayoutKind::HistoryFrame;
      block.length = bnode["length"].as<size_t>();
      if (block.length == 0) {
        throw std::runtime_error("[PolicySlot:" + name_ +
                                 "] history_frame length must be > 0");
      }
      block.order = bnode["order"].as<std::string>("oldest_first");
      if (block.order != "oldest_first" && block.order != "newest_first") {
        throw std::runtime_error("[PolicySlot:" + name_ +
                                 "] history_frame order must be oldest_first/newest_first");
      }
      block.include_current = bnode["include_current"].as<bool>(false);
      block.dim = block.length * obs_dim_;

      const size_t needed_prev =
          block.include_current ? (block.length > 0 ? block.length - 1 : 0) : block.length;
      history_capacity_ = std::max(history_capacity_, needed_prev);
    } else if (kind == "current_terms") {
      block.kind = LayoutKind::CurrentTerms;
      YAML::Node terms = bnode["terms"];
      if (!terms || !terms.IsSequence() || terms.size() == 0) {
        throw std::runtime_error("[PolicySlot:" + name_ +
                                 "] current_terms requires non-empty terms sequence");
      }
      for (auto tnode : terms) {
        const std::string tname = tnode.as<std::string>();
        auto it = obs_term_indices_.find(tname);
        if (it == obs_term_indices_.end()) {
          throw std::runtime_error("[PolicySlot:" + name_ +
                                   "] current_terms unknown term: " + tname);
        }
        block.term_indices.push_back(it->second);
        block.dim += obs_terms_[it->second].dim;
      }
    } else {
      throw std::runtime_error("[PolicySlot:" + name_ +
                               "] unknown layout block kind: " + kind);
    }

    layout_input_dim_ += block.dim;
    layout_blocks_.push_back(std::move(block));
  }

  if (layout_input_dim_ != input_dim_) {
    throw std::runtime_error(
        "[PolicySlot:" + name_ + "] input_dim mismatch: layout expected=" +
        std::to_string(layout_input_dim_) + " cfg input_dim=" +
        std::to_string(input_dim_));
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

void PolicySlot::assembleObsFrame(const LeggedState& state,
                                  const unitree::common::Gamepad& gamepad,
                                  size_t loop_cnt, double ll_dt) {
  std::fill(obs_now_.begin(), obs_now_.end(), 0.0f);

  if (mimic_source_) {
    mimic_source_->step(state);
  }

  for (const auto& term : obs_terms_) {
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
      const float phase = loop_cnt * ll_dt / cycle_time;
      constexpr float kTwoPi = 6.28318530718f;
      v[0] = std::sin(kTwoPi * phase);
      v[1] = std::cos(kTwoPi * phase);

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
      v = {gamepad.ly, -gamepad.lx, -gamepad.rx};
      auto it = commands_.find("base_velocity");
      if (it != commands_.end()) it->second.process(v);

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
    std::copy(v.begin(), v.end(), obs_now_.begin() + term.offset);
  }
}

void PolicySlot::stackObsGlobal() {
  obs_hist_.push_back(obs_now_);
  while (obs_hist_.size() > stack_len_) obs_hist_.pop_front();
  while (obs_hist_.size() < stack_len_) obs_hist_.push_front(obs_hist_.front());

  size_t out = 0;
  if (stack_order_ == "newest_first") {
    for (size_t k = 0; k < stack_len_; ++k) {
      const auto& fr = obs_hist_[stack_len_ - 1 - k];
      std::copy(fr.begin(), fr.end(), input_buf_.begin() + out);
      out += obs_dim_;
    }
  } else {
    for (size_t k = 0; k < stack_len_; ++k) {
      const auto& fr = obs_hist_[k];
      std::copy(fr.begin(), fr.end(), input_buf_.begin() + out);
      out += obs_dim_;
    }
  }
}

void PolicySlot::assembleObsByLayout() {
  size_t out = 0;
  for (const auto& block : layout_blocks_) {
    if (block.kind == LayoutKind::CurrentFrame) {
      std::copy(obs_now_.begin(), obs_now_.end(), input_buf_.begin() + out);
      out += obs_dim_;
      continue;
    }

    if (block.kind == LayoutKind::CurrentTerms) {
      for (const size_t term_idx : block.term_indices) {
        const auto& term = obs_terms_[term_idx];
        std::copy(obs_now_.begin() + term.offset,
                  obs_now_.begin() + term.offset + term.dim,
                  input_buf_.begin() + out);
        out += term.dim;
      }
      continue;
    }

    std::vector<const std::vector<float>*> timeline;
    timeline.reserve(obs_hist_.size() + 1);
    for (const auto& fr : obs_hist_) timeline.push_back(&fr);
    if (block.include_current) timeline.push_back(&obs_now_);

    const size_t take = std::min(block.length, timeline.size());
    const size_t pad = block.length - take;

    const std::vector<float>* pad_frame = &zeros_frame_;
    if (history_warmup_ == "repeat_first" && !timeline.empty()) {
      pad_frame = timeline.front();
    }

    std::vector<const std::vector<float>*> selected;
    selected.reserve(block.length);
    for (size_t i = 0; i < pad; ++i) selected.push_back(pad_frame);
    for (size_t i = timeline.size() - take; i < timeline.size(); ++i) {
      selected.push_back(timeline[i]);
    }

    if (block.order == "newest_first") {
      for (size_t i = 0; i < selected.size(); ++i) {
        const auto* fr = selected[selected.size() - 1 - i];
        std::copy(fr->begin(), fr->end(), input_buf_.begin() + out);
        out += obs_dim_;
      }
    } else {
      for (const auto* fr : selected) {
        std::copy(fr->begin(), fr->end(), input_buf_.begin() + out);
        out += obs_dim_;
      }
    }
  }

  if (out != input_buf_.size()) {
    throw std::runtime_error("[PolicySlot:" + name_ +
                             "] internal error: layout packed dim mismatch");
  }
}

void PolicySlot::pushObsHistory() {
  if (history_capacity_ == 0) return;
  obs_hist_.push_back(obs_now_);
  while (obs_hist_.size() > history_capacity_) {
    obs_hist_.pop_front();
  }
}

void PolicySlot::updatePolicy(const LeggedState& state,
                              const unitree::common::Gamepad& gamepad,
                              size_t loop_cnt, double ll_dt) {
  assembleObsFrame(state, gamepad, loop_cnt, ll_dt);

  if (use_layout_) {
    assembleObsByLayout();
    pushObsHistory();
  } else if (stack_len_ == 1) {
    std::copy(obs_now_.begin(), obs_now_.end(), input_buf_.begin());
  } else {
    stackObsGlobal();
  }

  policy_runner_->infer(input_buf_.data(), output_buf_.data());
  last_action_ = output_buf_;

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
  auto it = obs_term_indices_.find(name);
  if (it == obs_term_indices_.end()) {
    throw std::runtime_error("[PolicySlot:" + name_ + "] obs term not found: " + name);
  }
  return obs_terms_[it->second];
}

std::vector<std::string> PolicySlot::loadMimicTerms(const YAML::Node& params) {
  if (params["terms"] && params["terms"].IsSequence()) {
    return params["terms"].as<std::vector<std::string>>();
  }
  return {"joint_pos", "joint_vel", "motion_anchor_ori_b"};
}

} // namespace legged_rl_deploy
