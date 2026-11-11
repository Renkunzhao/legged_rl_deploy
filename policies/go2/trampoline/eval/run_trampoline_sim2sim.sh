#!/usr/bin/env bash
# Start an isolated loopback simulator/controller, evaluate, and stop both.
set -euo pipefail
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$EVAL_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO" ]]; then
  REPO="$(cd "$EVAL_DIR/../../../.." && pwd)"
fi
WS="$(cd "$REPO/../.." && pwd)"
SIM_CONFIG="$WS/src/unitree_mujoco/simulate/config/go2.yaml"
SIM_EVAL="$WS/src/unitree_mujoco/simulate/config/eval_trampoline.yaml"
POLICY_CONFIG="src/legged_rl_deploy/policies/go2/trampoline/config.yaml"
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --sim-config) SIM_CONFIG="$(realpath "$2")"; shift 2 ;;
    --sim-eval) SIM_EVAL="$(realpath "$2")"; shift 2 ;;
    --policy-config) POLICY_CONFIG="$(realpath "$2")"; shift 2 ;;
    --) shift; break ;;
    *) echo "Usage: $0 --output NEW_DIRECTORY [--sim-config YAML] [--sim-eval YAML] [--policy-config YAML] -- [evaluator options]" >&2; exit 2 ;;
  esac
done
[[ -n "$OUTPUT" && ! -e "$OUTPUT" ]] || { echo "--output must be a new directory" >&2; exit 2; }
mkdir -p "$OUTPUT"
OUTPUT="$(realpath "$OUTPUT")"
set +u
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
source "$WS/install/setup.bash"
set -u
export WORKSPACE="$WS" NetworkInterface=lo
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-83}"
[[ "$ROS_DOMAIN_ID" =~ ^[0-9]+$ ]] && ((ROS_DOMAIN_ID >= 1 && ROS_DOMAIN_ID <= 232)) || exit 2
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" multicast="false" /></Interfaces></General></Domain></CycloneDDS>'
export LD_LIBRARY_PATH="$WS/src/legged_rl_deploy/thirdparty/libtorch/lib:${LD_LIBRARY_PATH:-}"
cd "$WS"
cp "$SIM_CONFIG" "$OUTPUT/simulator.yaml"
cp "$SIM_EVAL" "$OUTPUT/simulator_eval.yaml"
cp "$POLICY_CONFIG" "$OUTPUT/policy.yaml"
/usr/bin/python3 - "$POLICY_CONFIG" "$OUTPUT" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys, yaml
policy, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
workspace = pathlib.Path(os.environ['WORKSPACE'])
config = yaml.safe_load(policy.read_text())
model = workspace / config['policy']['model_path']
revisions = {}
for name in ('legged_rl_deploy', 'unitree_lowlevel', 'unitree_mujoco'):
    repo = workspace / 'src' / name
    revisions[name] = {
        'head': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'status': subprocess.check_output(['git', '-C', str(repo), 'status', '--short'], text=True),
    }
manifest = {'model': str(model), 'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
            'repositories': revisions}
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
PY
PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill -INT -- "-$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid ros2 run unitree_mujoco unitree_mujoco --config "$SIM_CONFIG" --eval "$SIM_EVAL" --no-viewer --no-joystick > "$OUTPUT/simulator.log" 2>&1 &
PIDS+=("$!")
setsid ros2 run legged_rl_deploy legged_rl_deploy_node lo "$POLICY_CONFIG" --ros-args -p evaluation_mode:=true < /dev/null > "$OUTPUT/controller.log" 2>&1 &
PIDS+=("$!")
/usr/bin/python3 "$REPO/policies/go2/trampoline/eval/eval_trampoline_sim2sim.py" --output "$OUTPUT/evaluation" "$@" | tee "$OUTPUT/evaluator.log"
