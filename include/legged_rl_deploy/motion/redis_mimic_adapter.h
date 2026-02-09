#pragma once

#include <string>
#include <vector>

#include "legged_rl_deploy/motion/mimic_source.h"

#ifdef USE_HIREDIS
struct redisContext;
#endif

namespace legged_rl_deploy {

class RedisMimicAdapter final : public IMimicSource {
public:
  struct Config {
    std::string host = "127.0.0.1";
    int port = 6379;
    int db = 0;
    std::string key;
    int timeout_ms = 5;
    std::string fallback = "hold_last"; // hold_last | zeros | error
    std::vector<float> init;
  };

  RedisMimicAdapter(Config cfg, size_t output_dim);
  ~RedisMimicAdapter() override;

  void reset(const LeggedState& state, float policy_dt) override;
  void step(const LeggedState& state) override;
  void read(std::vector<float>& out) override;
  size_t dim() const override { return output_dim_; }

private:
  enum class FallbackMode { HoldLast, Zeros, Error };

  static FallbackMode parseFallbackMode(const std::string& mode);
  static std::vector<float> parseJsonArray(const std::string& payload);

  void connect();
  void applyFallback(std::vector<float>& out, const std::string& reason) const;
  void warn(const std::string& msg) const;

  Config cfg_;
  size_t output_dim_ = 0;
  FallbackMode fallback_mode_ = FallbackMode::HoldLast;
  std::vector<float> last_good_;
  mutable size_t warn_counter_ = 0;

#ifdef USE_HIREDIS
  ::redisContext* redis_ctx_ = nullptr;
#endif
};

} // namespace legged_rl_deploy
