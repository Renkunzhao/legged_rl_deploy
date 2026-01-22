#include "legged_rl_deploy/policy/policy_factory.h"

#include <stdexcept>

#ifdef USE_TORCH
#include "legged_rl_deploy/policy/torch_policy_runner.h"
#endif

#ifdef USE_ONNX
#include "legged_rl_deploy/policy/ort_policy_runner.h"
#endif

namespace legged_rl_deploy {

static std::string normalizeBackend(std::string s) {
  for (auto& c : s) c = static_cast<char>(::tolower(c));
  if (s == "libtorch") s = "torch";
  if (s == "onnxruntime") s = "onnx";
  return s;
}

std::unique_ptr<IPolicyRunner> makePolicyRunner(const std::string& backend_in) {
  const auto backend = normalizeBackend(backend_in);

  if (backend == "torch") {
#ifdef USE_TORCH
    return std::make_unique<TorchPolicyRunner>();
#else
    throw std::runtime_error("backend=torch requested, but built without USE_TORCH");
#endif
  }

  if (backend == "onnx") {
#ifdef USE_ONNX
    return std::make_unique<OrtPolicyRunner>();
#else
    throw std::runtime_error("backend=onnx requested, but built without USE_ONNX");
#endif
  }

  throw std::runtime_error("Unknown policy backend: " + backend_in);
}

}  // namespace legged_rl_deploy
