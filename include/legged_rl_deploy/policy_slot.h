#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "legged_rl_deploy/motion/mimic_source.h"
#include "legged_rl_deploy/policy/i_policy_runner.h"
#include "legged_rl_deploy/processor.h"

#include <legged_base/LeggedModel.h>
#include <legged_base/LeggedState.h>
#include <unitree_lowlevel/gamepad.hpp>

namespace legged_rl_deploy {

class PolicySlot {
public:
  PolicySlot(const std::string& name, const YAML::Node& policyNode,
             const LeggedModel& model);

  void init();
  void reset(const LeggedState& state);
  void update(const LeggedState& state, const unitree::common::Gamepad& gamepad,
              size_t loop_cnt, double ll_dt);

  const std::string& name() const { return name_; }
  float policyDt() const { return policy_dt_; }
  size_t outputDim() const { return output_dim_; }
  const std::vector<float>& outputBuf() const { return output_buf_; }
  const std::vector<size_t>& jointIdsMap() const { return joint_ids_map_; }
  const std::vector<float>& stiffness() const { return stiffness_; }
  const std::vector<float>& damping() const { return damping_; }

private:
  struct ObsTerm {
    std::string name;
    size_t dim = 0;
    size_t offset = 0;
    std::optional<Processor> proc;
    YAML::Node params;
  };

  enum class LayoutKind { CurrentFrame, HistoryFrame, CurrentTerms };

  struct ObsLayoutBlock {
    LayoutKind kind = LayoutKind::CurrentFrame;
    size_t length = 0; // history_frame only
    std::string order = "oldest_first";
    bool include_current = false;
    std::vector<size_t> term_indices;
    size_t dim = 0;
  };

  void registryObsTerms(YAML::Node node);
  void calculateObsTerm(ObsTerm& term);
  void parseObsLayout(const YAML::Node& observations);
  void initMimicSource();
  void assembleObsFrame(const LeggedState& state,
                        const unitree::common::Gamepad& gamepad,
                        size_t loop_cnt, double ll_dt);
  void stackObsGlobal();
  void assembleObsByLayout();
  void pushObsHistory();
  void updatePolicy(const LeggedState& state,
                    const unitree::common::Gamepad& gamepad, size_t loop_cnt,
                    double ll_dt);
  const ObsTerm& getObsTermByName(const std::string& name) const;
  static std::vector<std::string> loadMimicTerms(const YAML::Node& params);

  std::string name_;
  YAML::Node policyNode_;
  const LeggedModel& robot_model_;

  std::unique_ptr<IPolicyRunner> policy_runner_;
  size_t input_dim_ = 0;
  size_t output_dim_ = 0;
  std::vector<float> input_buf_;
  std::vector<float> output_buf_;

  float policy_dt_ = 0.02f;
  std::vector<size_t> joint_ids_map_;
  std::vector<float> stiffness_;
  std::vector<float> damping_;
  std::vector<float> last_action_;

  std::unordered_map<std::string, Processor> commands_;
  std::unordered_map<std::string, Processor> actions_;

  std::vector<ObsTerm> obs_terms_;
  std::unordered_map<std::string, size_t> obs_term_indices_;

  size_t stack_len_ = 1;
  std::string stack_order_ = "oldest_first";
  size_t obs_dim_ = 0;
  std::vector<float> obs_now_;
  std::deque<std::vector<float>> obs_hist_;
  std::vector<float> zeros_frame_;

  bool use_layout_ = false;
  std::vector<ObsLayoutBlock> layout_blocks_;
  size_t history_capacity_ = 0;
  std::string history_warmup_ = "repeat_first";
  size_t layout_input_dim_ = 0;

  bool has_mimic_term_ = false;
  YAML::Node mimic_params_;
  std::unique_ptr<IMimicSource> mimic_source_;
};

} // namespace legged_rl_deploy
