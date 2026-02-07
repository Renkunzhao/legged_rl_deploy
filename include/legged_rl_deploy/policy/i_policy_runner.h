#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace legged_rl_deploy {

class IPolicyRunner {
public:
  virtual ~IPolicyRunner() = default;

  virtual void load(const std::string& model_path, int input_dim, int output_dim) = 0;

  // 输入输出 float32，caller 提供 buffer（不在 infer 内部分配）
  virtual void infer(const float* input, float* output) = 0;

  /// Set an extra named input tensor (beyond the primary obs).
  /// Call before infer(). The runner keeps a reference to `data` until the
  /// next infer() call, so the caller must keep the buffer alive.
  virtual void setExtraInput(const std::string& /*name*/,
                             const float* /*data*/,
                             const std::vector<int64_t>& /*shape*/) {}

protected:
  int input_dim_, output_dim_;
};

}  // namespace legged_rl_deploy
