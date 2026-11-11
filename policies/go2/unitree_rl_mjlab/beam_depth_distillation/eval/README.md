# Automated beam (bridge) sim2sim evaluation

The simulator's C++ `beam` episode monitor decides each trial's outcome
(`SUCCESS` / `SELF_COLLISION` / `ILLEGAL_CONTACT` / `FELL_OVER` / `TIMEOUT`).
The Python evaluator only drives the episode loop and aggregates results.

```bash
cd /home/rkz/code/unitree_ws
source src/unitree_lowlevel/scripts/setup.sh lo jazzy
export ROS_DOMAIN_ID=1
```

## Simulator

Use the beam task config: it enables `episode_monitor: beam`, episode services,
inertial ground truth, and payload resampling.

```bash
ros2 run unitree_mujoco unitree_mujoco \
  --config src/unitree_mujoco/simulate/config/go2.yaml \
  --eval src/unitree_mujoco/simulate/config/eval_beam.yaml --no-joystick

# headless
ros2 run unitree_mujoco unitree_mujoco \
  --config src/unitree_mujoco/simulate/config/go2.yaml \
  --eval src/unitree_mujoco/simulate/config/eval_beam.yaml --no-viewer --no-joystick
```

## Depth preprocessor

```bash
ros2 run legged_rl_deploy depth_image_preprocessor_node.py \
  --ros-args --params-file \
  src/legged_rl_deploy/policies/go2/unitree_rl_mjlab/beam_depth_distillation/depth_image_preprocessor.yaml
```

Check the topics if the policy looks blind:

```bash
ros2 topic hz /camera/depth/image_rect_raw --window 300
ros2 topic hz /unitree_go2_beam_depth/depth_m --window 300
```

## Controller

```bash
ros2 run legged_rl_deploy legged_rl_deploy_node lo \
  src/legged_rl_deploy/policies/go2/unitree_rl_mjlab/beam_depth_distillation/config_explicit.yaml \
  --ros-args -p evaluation_mode:=true
```

## Evaluator

```bash
python3 src/legged_rl_deploy/policies/go2/unitree_rl_mjlab/beam_depth_distillation/eval/eval_beam_sim2sim.py \
  --trials 64 \
  --velocity 0.6 \
  --seed 42 \
  --reset-position-jitter-m 0.05 \
  --reset-yaw-jitter-rad 0.10
```

The evaluator requires `--no-joystick` on the simulator (it asserts that only
itself publishes `/wirelesscontroller`) and dumps one JSONL row per trial plus a
`*.summary.json`.
