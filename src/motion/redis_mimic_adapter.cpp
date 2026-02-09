#include "legged_rl_deploy/motion/redis_mimic_adapter.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

#ifdef USE_HIREDIS
#include <hiredis/hiredis.h>
#endif

#include <unitree_lowlevel/gamepad_fsm.hpp>

namespace legged_rl_deploy {

RedisMimicAdapter::RedisMimicAdapter(Config cfg, size_t output_dim)
    : cfg_(std::move(cfg)), output_dim_(output_dim) {
  if (cfg_.key.empty()) {
    throw std::runtime_error("[RedisMimicAdapter] redis.key is required");
  }
  if (output_dim_ == 0) {
    throw std::runtime_error("[RedisMimicAdapter] output dim must be positive");
  }
  fallback_mode_ = parseFallbackMode(cfg_.fallback);
  last_good_.assign(output_dim_, 0.0f);

  if (!cfg_.init.empty()) {
    if (cfg_.init.size() != output_dim_) {
      throw std::runtime_error("[RedisMimicAdapter] init dim mismatch");
    }
    last_good_ = cfg_.init;
  }

  connect();

  if (!cfg_.motion_start_trigger.empty()) {
    const auto parsed = unitree::common::GamepadFSM::parseTrigger(
        "motion_start_signal", cfg_.motion_start_trigger);
    motion_start_modifiers_ = parsed.modifiers;
    motion_start_button_ = parsed.trigger;
    motion_start_enabled_ = !motion_start_button_.empty();
    std::cout << "[RedisMimicAdapter] motion start trigger: "
              << cfg_.motion_start_trigger << std::endl;
  }
}

RedisMimicAdapter::~RedisMimicAdapter() {
#ifdef USE_HIREDIS
  if (redis_ctx_) {
    redisFree(redis_ctx_);
    redis_ctx_ = nullptr;
  }
#endif
}

void RedisMimicAdapter::reset(const LeggedState&, float) {}

void RedisMimicAdapter::step(const LeggedState&) {}

void RedisMimicAdapter::onGamepad(const unitree::common::Gamepad& gamepad) {
#ifdef USE_HIREDIS
  if (!motion_start_enabled_ || !redis_ctx_) return;

  for (const auto& mod : motion_start_modifiers_) {
    if (!unitree::common::GamepadFSM::buttonPressed(gamepad, mod)) {
      return;
    }
  }
  if (!unitree::common::GamepadFSM::buttonOnPress(gamepad, motion_start_button_)) {
    return;
  }

  redisReply* reply = static_cast<redisReply*>(
      redisCommand(redis_ctx_, "SET motion_start_signal 1"));
  if (!reply) {
    warn("failed to send motion_start_signal");
    return;
  }
  const std::unique_ptr<redisReply, decltype(&freeReplyObject)> guard(
      reply, freeReplyObject);
  if (reply->type == REDIS_REPLY_ERROR) {
    warn("motion_start_signal SET error");
    return;
  }

  std::cout << "[RedisMimicAdapter] motion_start_signal=1" << std::endl;
#else
  (void)gamepad;
#endif
}

void RedisMimicAdapter::read(std::vector<float>& out) {
#ifdef USE_HIREDIS
  redisReply* reply = static_cast<redisReply*>(
      redisCommand(redis_ctx_, "GET %s", cfg_.key.c_str()));

  if (!reply) {
    applyFallback(out, "redis GET returned null");
    return;
  }

  const std::unique_ptr<redisReply, decltype(&freeReplyObject)> guard(
      reply, freeReplyObject);

  if (reply->type != REDIS_REPLY_STRING || reply->str == nullptr) {
    applyFallback(out, "redis key missing or not string");
    return;
  }

  std::vector<float> parsed;
  try {
    parsed = parseJsonArray(reply->str);
  } catch (const std::exception& e) {
    applyFallback(out, std::string("json parse failed: ") + e.what());
    return;
  }

  if (parsed.size() != output_dim_) {
    applyFallback(out, "dim mismatch from redis payload");
    return;
  }

  last_good_ = parsed;
  out = last_good_;
#else
  (void)out;
  throw std::runtime_error(
      "[RedisMimicAdapter] redis source requires hiredis (build with USE_HIREDIS)");
#endif
}

RedisMimicAdapter::FallbackMode RedisMimicAdapter::parseFallbackMode(
    const std::string& mode) {
  if (mode == "hold_last") return FallbackMode::HoldLast;
  if (mode == "zeros") return FallbackMode::Zeros;
  if (mode == "error") return FallbackMode::Error;
  throw std::runtime_error("[RedisMimicAdapter] unsupported fallback mode: " + mode);
}

std::vector<float> RedisMimicAdapter::parseJsonArray(const std::string& payload) {
  std::vector<float> values;
  const char* p = payload.c_str();
  const char* end = p + payload.size();

  while (p < end) {
    while (p < end && (std::isspace(*p) || *p == '[' || *p == ']' || *p == ',')) {
      ++p;
    }
    if (p >= end) break;

    errno = 0;
    char* parse_end = nullptr;
    const float v = std::strtof(p, &parse_end);
    if (parse_end == p || errno == ERANGE) {
      throw std::runtime_error("invalid float token in payload");
    }
    values.push_back(v);
    p = parse_end;
  }

  if (values.empty()) {
    throw std::runtime_error("empty vector payload");
  }
  return values;
}

void RedisMimicAdapter::connect() {
#ifdef USE_HIREDIS
  struct timeval timeout;
  timeout.tv_sec = cfg_.timeout_ms / 1000;
  timeout.tv_usec = (cfg_.timeout_ms % 1000) * 1000;

  redis_ctx_ = redisConnectWithTimeout(cfg_.host.c_str(), cfg_.port, timeout);
  if (!redis_ctx_ || redis_ctx_->err) {
    const std::string err = redis_ctx_ ? redis_ctx_->errstr : "unknown error";
    throw std::runtime_error("[RedisMimicAdapter] connect failed: " + err);
  }

  if (cfg_.db != 0) {
    redisReply* reply =
        static_cast<redisReply*>(redisCommand(redis_ctx_, "SELECT %d", cfg_.db));
    if (!reply) {
      throw std::runtime_error("[RedisMimicAdapter] SELECT failed");
    }
    const std::unique_ptr<redisReply, decltype(&freeReplyObject)> guard(
        reply, freeReplyObject);
    if (reply->type == REDIS_REPLY_ERROR) {
      throw std::runtime_error("[RedisMimicAdapter] SELECT failed: " +
                               std::string(reply->str ? reply->str : ""));
    }
  }
#else
  throw std::runtime_error(
      "[RedisMimicAdapter] redis source requires hiredis (build with USE_HIREDIS)");
#endif
}

void RedisMimicAdapter::applyFallback(std::vector<float>& out,
                                      const std::string& reason) const {
  if (fallback_mode_ == FallbackMode::Error) {
    throw std::runtime_error("[RedisMimicAdapter] " + reason);
  }
  if (fallback_mode_ == FallbackMode::Zeros) {
    out.assign(output_dim_, 0.0f);
    warn(reason + ", fallback=zeros");
    return;
  }

  out = last_good_;
  warn(reason + ", fallback=hold_last");
}

void RedisMimicAdapter::warn(const std::string& msg) const {
  warn_counter_++;
  if ((warn_counter_ % 200) == 1) {
    std::cerr << "[RedisMimicAdapter] " << msg << std::endl;
  }
}

} // namespace legged_rl_deploy
