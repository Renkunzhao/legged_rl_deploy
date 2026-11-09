# Automated Beam Sim-to-Sim Evaluation

```bash
cd /home/rkz/code/unitree_ws
source src/unitree_lowlevel/scripts/setup.sh lo jazzy
export ROS_DOMAIN_ID=1
```

## Sim
```bash
ros2 run unitree_mujoco unitree_mujoco --no-joystick

ros2 run unitree_mujoco unitree_mujoco --no-viewer --no-joystick

ros2 run unitree_mujoco unitree_mujoco \
 --no-joystick 2>&1 | tee sim_beam.log
```

## Preprocessor
```bash
ros2 run legged_rl_deploy depth_image_preprocessor_node.py \
  --ros-args --params-file \
  src/legged_rl_deploy/policies/go2/unitree_rl_mjlab/beam_depth_distillation/depth_image_preprocessor.yaml
```

## Record
```bash
ros2 topic hz /camera/depth/image_rect_raw --window 300
ros2 topic hz /unitree_go2_beam_depth/depth_m --window 300

ros2 run plotjuggler plotjuggler
```

## Controller
```bash
ros2 run legged_rl_deploy legged_rl_deploy_node lo \
  src/legged_rl_deploy/policies/go2/unitree_rl_mjlab/beam_depth_distillation/config_explicit.yaml \
  --ros-args -p evaluation_mode:=true
```

## Eval
```bash
python3 src/legged_rl_deploy/scripts/eval_beam_sim2sim.py \
  --trials 64 \
  --velocity 0.6 \
  --seed 42 \
  --reset-position-jitter-m 0.05 \
  --reset-yaw-jitter-rad 0.10
```
