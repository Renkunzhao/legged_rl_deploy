#include "legged_rl_deploy/policy/policy_factory.h"

#include <stdexcept>

#ifdef USE_TORCH
#include "legged_rl_deploy/policy/torch_policy_runner.h"
#endif

#ifdef USE_ORT
#include "legged_rl_deploy/policy/ort_policy_runner.h"
#endif

namespace legged_rl_deploy {

static std::string normalizeBackend(std::string s) {
  for (auto& c : s) c = static_cast<char>(::tolower(c));
  if (s == "torch") s = "libtorch";
  if (s == "ort") s = "onnxruntime";
  return s;
}

std::unique_ptr<IPolicyRunner> makePolicyRunner(const std::string& backend_in) {
  const auto backend = normalizeBackend(backend_in);

  if (backend == "libtorch") {
#ifdef USE_TORCH
    return std::make_unique<TorchPolicyRunner>();
#else
    throw std::runtime_error("backend=libtorch requested, but built without USE_TORCH");
#endif
  }

  if (backend == "onnxruntime") {
#ifdef USE_ORT
    return std::make_unique<OrtPolicyRunner>();
#else
    throw std::runtime_error("backend=onnxruntime requested, but built without USE_ORT");
#endif
  }

  throw std::runtime_error("Unknown policy backend: " + backend_in);
}

}  // namespace legged_rl_deploy
