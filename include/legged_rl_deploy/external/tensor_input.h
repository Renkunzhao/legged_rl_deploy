#pragma once

#include <vector>

namespace legged_rl_deploy {

class TensorInput {
public:
  virtual ~TensorInput() = default;
  virtual void read(std::vector<float>& destination) const = 0;
};

}  // namespace legged_rl_deploy
