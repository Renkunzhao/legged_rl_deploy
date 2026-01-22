#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "legged_rl_deploy/deploy_config.h"
#include "legged_rl_deploy/observation_assembler.h"

namespace legged_rl_deploy {

// Given ObsTerm config, return raw observation (float)
using ObsFn = std::function<std::vector<double>(const ObsTerm&)>;

/**
 * ObservationRegistry
 *
 * - Registers observation generators by name
 * - Automatically fills ObservationAssembler according to YAML order
 * - Special-cases "last_action" as policy feedback
 */
class ObservationRegistry {
public:
  void Register(const std::string& name, ObsFn fn) {
    registry_[name] = std::move(fn);
  }

  bool Has(const std::string& name) const {
    return registry_.count(name) != 0;
  }

  /**
   * Fill assembler according to cfg.observations.terms order.
   *
   * Rules:
   * - If term name == "last_action": use last_action directly
   * - Otherwise, must have a registered ObsFn
   */
  void Fill(const ObservationsConfig& cfg,
            ObservationAssembler& assembler,
            const std::vector<float>& last_action) const {
    for (const auto& kv : cfg.terms) {
      const std::string& name = kv.first;
      const ObsTerm& term = kv.second;

      if (name == "last_action") {
        assembler.PushProcessed(name, last_action);
        continue;
      }

      auto it = registry_.find(name);
      if (it == registry_.end()) {
        throw std::runtime_error(
            "ObservationRegistry: no generator registered for term '" + name + "'.");
      }

      assembler.PushRaw(name, it->second(term));
    }
  }

private:
  std::unordered_map<std::string, ObsFn> registry_;
};

}  // namespace legged_rl_deploy
