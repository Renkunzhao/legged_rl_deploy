#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "legged_rl_deploy/motion/mimic_source.h"

#include <legged_base/LeggedModel.h>

namespace legged_rl_deploy {

struct IMotionEvaluator {
  virtual ~IMotionEvaluator() = default;
  virtual void evaluate(float time_step, Eigen::VectorXf& joint_pos,
                        Eigen::VectorXf& joint_vel,
                        Eigen::Quaternionf& root_quat) = 0;
};

std::unique_ptr<IMotionEvaluator> makeOnnxMotionEvaluator(
    const std::string& model_path);

class LocalMimicAdapter final : public IMimicSource {
public:
  struct Config {
    std::string file;
    float fps = 50.0f;
    float time_start = 0.0f;
    float time_end = -1.0f;
    bool hardware_order = true;
    std::vector<std::string> terms = {
        "joint_pos", "joint_vel", "motion_anchor_ori_b"};
  };

  LocalMimicAdapter(Config cfg, const LeggedModel& robot_model,
                    std::vector<size_t> joint_ids_map, size_t output_dim);

  void reset(const LeggedState& state, float policy_dt) override;
  void step(const LeggedState& state) override;
  void read(std::vector<float>& out) override;
  size_t dim() const override { return output_dim_; }

private:
  void loadDataFromNpz(const std::string& motion_file);
  void loadDataFromCsv(const std::string& motion_file);
  void update(float time);
  void buildOutput(const LeggedState& state);

  Eigen::VectorXf jointPosRaw() const;
  Eigen::VectorXf jointVelRaw() const;
  Eigen::Quaternionf rootQuaternion() const;

  Eigen::VectorXf jointPosTrainingOrder() const;
  Eigen::VectorXf jointVelTrainingOrder() const;
  Eigen::VectorXf jointPosUrdfOrder() const;

  static bool endsWith(const std::string& s, const std::string& suffix);
  static std::vector<std::vector<float>> loadCsv(const std::string& filename);
  std::vector<Eigen::VectorXf> computeRawDerivative(
      const std::vector<Eigen::VectorXf>& data) const;
  size_t inferOutputDim() const;

  Config cfg_;
  const LeggedModel& robot_model_;
  std::vector<size_t> joint_ids_map_;
  size_t output_dim_ = 0;

  float motion_dt_ = 0.02f;
  int num_frames_ = 0;
  float duration_ = 0.0f;
  int frame_ = 0;

  std::vector<Eigen::Vector3f> root_positions_;
  std::vector<Eigen::Quaternionf> root_quaternions_;
  std::vector<Eigen::VectorXf> dof_positions_;
  std::vector<Eigen::VectorXf> dof_velocities_;

  float time_start_ = 0.0f;
  float time_end_ = 0.0f;
  float policy_dt_ = 0.02f;
  size_t step_cnt_ = 0;
  Eigen::Quaternionf yaw_align_ = Eigen::Quaternionf::Identity();

  std::unique_ptr<IMotionEvaluator> onnx_eval_;
  Eigen::VectorXf onnx_joint_pos_;
  Eigen::VectorXf onnx_joint_vel_;
  Eigen::Quaternionf onnx_root_quat_ = Eigen::Quaternionf::Identity();

  std::vector<float> last_output_;
};

} // namespace legged_rl_deploy
