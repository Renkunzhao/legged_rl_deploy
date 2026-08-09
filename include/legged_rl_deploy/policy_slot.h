#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "legged_rl_deploy/external/ros_image_tensor_input.h"
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
             const LeggedModel& model, rclcpp::Node& node);

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
  bool hasValidOutput() const { return has_valid_output_; }

private:
  enum class OrderMode { OldestFirst, NewestFirst, Explicit };
  enum class HistoryWarmupMode { RepeatFirst, Zero };

  struct ObsOrderSpec {
    OrderMode mode = OrderMode::OldestFirst;
    std::vector<size_t> lags;
  };

  struct ObsTerm {
    std::string name;
    size_t dim = 0;
    size_t offset = 0;
    std::optional<Processor> proc;
    YAML::Node params;
    size_t default_length = 1;
    bool has_default_length = false;
    ObsOrderSpec default_order;
    HistoryWarmupMode default_history_warmup = HistoryWarmupMode::RepeatFirst;
  };

  struct AssembleBlock {
    std::vector<size_t> term_indices;
    ObsOrderSpec order;
    size_t length = 1;
    bool has_length = false;
    HistoryWarmupMode history_warmup = HistoryWarmupMode::RepeatFirst;
    std::vector<size_t> lags;
    size_t dim = 0;
  };

  void registryObsTerms(YAML::Node node);
  void calculateObsTerm(ObsTerm& term);
  void parseAssemble(const YAML::Node& observations);
  void computeTermHistoryCapacities();
  void initMimicSource();
  void initExternalInputs();
  void updateVelocityCommand(const unitree::common::Gamepad& gamepad);
  void assembleObsFrame(const LeggedState& state,
                        const unitree::common::Gamepad& gamepad,
                        size_t loop_cnt, double ll_dt);
  void assembleDefaultTermMajor();
  void assembleFromBlocks();
  void pushTermHistory();
  void updatePolicy(const LeggedState& state,
                    const unitree::common::Gamepad& gamepad, size_t loop_cnt,
                    double ll_dt);
  const ObsTerm& getObsTermByName(const std::string& name) const;
  static ObsOrderSpec parseOrderSpec(const YAML::Node& node);
  static std::vector<size_t> resolveLags(const ObsOrderSpec& spec, size_t length);
  static HistoryWarmupMode parseWarmup(const YAML::Node& node,
                                       HistoryWarmupMode default_mode);
  const std::vector<float>& sampleTermAtLag(size_t term_idx, size_t lag,
                                            HistoryWarmupMode warmup) const;
  static std::vector<std::string> loadMimicTerms(const YAML::Node& params);

  std::string name_;
  YAML::Node policyNode_;
  const LeggedModel& robot_model_;
  rclcpp::Node& node_;

  std::unique_ptr<IPolicyRunner> policy_runner_;
  size_t input_dim_ = 0;
  size_t output_dim_ = 0;
  std::vector<float> input_buf_;
  std::vector<float> output_buf_;
  std::vector<float> raw_output_;
  std::vector<RuntimeTensor> runtime_inputs_;
  std::unordered_map<std::string, std::vector<float>> external_input_buffers_;
  std::unordered_map<std::string, std::unique_ptr<RosImageTensorInput>>
      external_inputs_;

  float policy_dt_ = 0.02f;
  std::vector<size_t> joint_ids_map_;
  std::vector<float> stiffness_;
  std::vector<float> damping_;
  std::vector<float> last_action_;

  std::unordered_map<std::string, Processor> commands_;
  std::unordered_map<std::string, Processor> actions_;
  std::vector<float> velocity_command_{0.0f, 0.0f, 0.0f};
  std::vector<float> velocity_rate_limit_;

  std::vector<ObsTerm> obs_terms_;
  std::unordered_map<std::string, size_t> obs_term_indices_;

  size_t obs_dim_ = 0;
  std::vector<float> obs_now_;
  std::vector<std::vector<float>> term_now_;
  std::vector<std::vector<float>> term_zero_;
  std::vector<std::deque<std::vector<float>>> term_hist_;
  std::vector<std::vector<size_t>> term_default_lags_;
  std::vector<size_t> term_history_capacity_;
  bool use_assemble_ = false;
  std::vector<AssembleBlock> assemble_blocks_;

  bool has_mimic_term_ = false;
  YAML::Node mimic_params_;
  std::unique_ptr<IMimicSource> mimic_source_;
  bool has_valid_output_ = false;
};

} // namespace legged_rl_deploy
