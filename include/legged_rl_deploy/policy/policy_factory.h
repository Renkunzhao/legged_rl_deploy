#pragma once
#include <memory>
#include <string>

#include "legged_rl_deploy/policy/i_policy_runner.h"

namespace legged_rl_deploy {

std::unique_ptr<IPolicyRunner> makePolicyRunner(const std::string& backend);

}  // namespace legged_rl_deploy
