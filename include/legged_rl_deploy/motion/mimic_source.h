#pragma once

#include <cstddef>
#include <vector>

#include <legged_base/LeggedState.h>

namespace legged_rl_deploy {

class IMimicSource {
public:
  virtual ~IMimicSource() = default;

  virtual void reset(const LeggedState& state, float policy_dt) = 0;
  virtual void step(const LeggedState& state) = 0;
  virtual void read(std::vector<float>& out) = 0;
  virtual size_t dim() const = 0;
};

} // namespace legged_rl_deploy
