#pragma once
#include <string>

namespace legged_rl_deploy {

class IPolicyRunner {
public:
  virtual ~IPolicyRunner() = default;

  virtual void load(const std::string& model_path, int input_dim, int output_dim) = 0;

  // 输入输出 float32，caller 提供 buffer（不在 infer 内部分配）
  virtual void infer(const float* input, float* output) = 0;

protected:
  int input_dim_, output_dim_;
};

}  // namespace legged_rl_deploy
