// motion_onnx_loader.cpp — On-demand motion evaluation via ONNX Runtime
//
// The ONNX model has two independent sub-graphs:
//   obs  → actions           (used by OrtPolicyRunner — ignored here)
//   time_step → joint_pos, joint_vel, body_quat_w, ...
//
// This evaluator only cares about the time_step→motion path.
// It feeds obs=0 (the paths are independent) and reads motion outputs.

#include "legged_rl_deploy/motion/motion_loader.h"

#ifdef USE_ORT
#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {

static inline int64_t numel(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (auto v : s) n *= (v > 0 ? v : 1);
  return n;
}

// -----------------------------------------------------------------------
// OnnxMotionEvaluator — holds an ORT session, evaluates motion on demand
// -----------------------------------------------------------------------
class OnnxMotionEvaluator final : public IMotionEvaluator {
public:
  explicit OnnxMotionEvaluator(const std::string& model_path) {
    opt_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    opt_.SetIntraOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opt_);

    Ort::AllocatorWithDefaultOptions alloc;

    // ---- inputs ----
    const size_t n_in = session_->GetInputCount();
    in_names_.resize(n_in);
    in_names_c_.resize(n_in);
    in_bufs_.resize(n_in);
    in_shapes_.resize(n_in);

    for (size_t i = 0; i < n_in; ++i) {
      auto n = session_->GetInputNameAllocated(i, alloc);
      in_names_[i] = n.get();
      in_names_c_[i] = in_names_[i].c_str();

      auto type_info = session_->GetInputTypeInfo(i);
      in_shapes_[i] = type_info.GetTensorTypeAndShapeInfo().GetShape();
      in_bufs_[i].assign(static_cast<size_t>(numel(in_shapes_[i])), 0.0f);

      if (in_names_[i] == "time_step") {
        ts_idx_ = i;
      }
    }

    if (ts_idx_ == SIZE_MAX) {
      throw std::runtime_error(
          "[OnnxMotionEvaluator] Model has no 'time_step' input");
    }

    // ---- outputs ----
    const size_t n_out = session_->GetOutputCount();
    out_names_.resize(n_out);
    out_names_c_.resize(n_out);

    for (size_t i = 0; i < n_out; ++i) {
      auto n = session_->GetOutputNameAllocated(i, alloc);
      out_names_[i] = n.get();
      out_names_c_[i] = out_names_[i].c_str();

      if (out_names_[i] == "joint_pos")   jp_idx_ = i;
      if (out_names_[i] == "joint_vel")   jv_idx_ = i;
      if (out_names_[i] == "body_quat_w") bq_idx_ = i;
    }

    if (jp_idx_ == SIZE_MAX || jv_idx_ == SIZE_MAX) {
      throw std::runtime_error(
          "[OnnxMotionEvaluator] Model must have 'joint_pos' and 'joint_vel' outputs");
    }

    // ---- self-check: run once at t=0 ----
    run(0.0f);
    std::cout << "[OnnxMotionEvaluator] Loaded: " << model_path
              << "  (nj=" << last_jp_.size() << ")" << std::endl;
  }

  void evaluate(float time_step,
                Eigen::VectorXf& joint_pos,
                Eigen::VectorXf& joint_vel,
                Eigen::Quaternionf& root_quat) override {
    run(time_step);
    joint_pos = last_jp_;
    joint_vel = last_jv_;
    root_quat = last_rq_;
  }

private:
  void run(float time_step) {
    in_bufs_[ts_idx_][0] = time_step;

    const size_t n_in  = in_names_.size();
    const size_t n_out = out_names_.size();

    std::vector<Ort::Value> tensors;
    tensors.reserve(n_in);
    for (size_t i = 0; i < n_in; ++i) {
      tensors.push_back(Ort::Value::CreateTensor<float>(
          mem_, in_bufs_[i].data(), in_bufs_[i].size(),
          in_shapes_[i].data(), in_shapes_[i].size()));
    }

    auto outs = session_->Run(
        Ort::RunOptions{nullptr},
        in_names_c_.data(), tensors.data(), n_in,
        out_names_c_.data(), n_out);

    // joint_pos [1, nj]
    {
      auto shape = outs[jp_idx_].GetTensorTypeAndShapeInfo().GetShape();
      int nj = static_cast<int>(shape.back());
      const float* p = outs[jp_idx_].GetTensorData<float>();
      last_jp_ = Eigen::Map<const Eigen::VectorXf>(p, nj);
    }
    // joint_vel [1, nj]
    {
      auto shape = outs[jv_idx_].GetTensorTypeAndShapeInfo().GetShape();
      int nj = static_cast<int>(shape.back());
      const float* p = outs[jv_idx_].GetTensorData<float>();
      last_jv_ = Eigen::Map<const Eigen::VectorXf>(p, nj);
    }
    // body_quat_w [1, num_bodies, 4] — body 0 = root, order: w,x,y,z
    if (bq_idx_ != SIZE_MAX) {
      const float* p = outs[bq_idx_].GetTensorData<float>();
      last_rq_ = Eigen::Quaternionf(p[0], p[1], p[2], p[3]);
    }
  }

  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "onnx_motion"};
  Ort::SessionOptions opt_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo mem_{
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

  std::vector<std::string>            in_names_;
  std::vector<const char*>            in_names_c_;
  std::vector<std::vector<float>>     in_bufs_;
  std::vector<std::vector<int64_t>>   in_shapes_;
  size_t ts_idx_ = SIZE_MAX;

  std::vector<std::string>  out_names_;
  std::vector<const char*>  out_names_c_;
  size_t jp_idx_ = SIZE_MAX;
  size_t jv_idx_ = SIZE_MAX;
  size_t bq_idx_ = SIZE_MAX;

  // Cached latest results
  Eigen::VectorXf last_jp_;
  Eigen::VectorXf last_jv_;
  Eigen::Quaternionf last_rq_ = Eigen::Quaternionf::Identity();
};

}  // anonymous namespace

std::unique_ptr<IMotionEvaluator> makeOnnxMotionEvaluator(
    const std::string& model_path) {
  return std::make_unique<OnnxMotionEvaluator>(model_path);
}

#else  // !USE_ORT

std::unique_ptr<IMotionEvaluator> makeOnnxMotionEvaluator(
    const std::string& /*model_path*/) {
  throw std::runtime_error(
      "ONNX motion loading requires ORT (compile with USE_ORT)");
}

#endif
