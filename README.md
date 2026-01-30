# Legged RL Deploy

A deployment framework for reinforcement learning (RL) policies on Unitree Go2 and G1 robots.

## Features

- Configurable via `config.yaml`
- Support for TorchScript and ONNX Runtime inference
- Multiple pretrained policies included

### Pretrained Policy Sources

| Source | Policies |
|--------|----------|
| [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | `go2_velocity_unitree` |
| [My_unitree_go2_gym](https://github.com/yusongmin1/My_unitree_go2_gym) | `go2_jump`, `go2_handstand`, `go2_legstand`, `go2_spring_jump`, `go2_trot` |
| [mjlab](https://github.com/mujocolab/mjlab) | `go2_velocity_mjlab` |

*Thanks to the authors of these projects for their contributions.*

## Video Demonstrations

- [Go2 Hop (Bilibili)](https://www.bilibili.com/video/BV1FTzfBcEL6/?vd_source=a178b41776a8e28bb0ca67b41f8f1fe8#reply287915981505)

## Installation

### Dependencies

- [TorchScript](TorchScript.md)

- ONNX Runtime (Coming soon)   .

- [unitree_lowlevel](https://github.com/Renkunzhao/unitree_lowlevel.git)

### Build

```bash
cd unitree_ws/src
git clone https://github.com/Renkunzhao/legged_rl_deploy.git
cd ..

# Ensure CMAKE_PREFIX_PATH and LD_LIBRARY_PATH include torch paths 
source install/setup.bash
colcon build --packages-up-to legged_rl_deploy 
```

```bash
source src/unitree_lowlevel/scripts/setup.sh <network-interface> $ROS_DISTRO
ros2 run legged_rl_deploy legged_rl_deploy_node $NetworkInterface $WORKSPACE/src/legged_rl_deploy/config/go2-trot.yaml
```

#### Troubleshooting

**CMake CUDA Dialect Error:**
```
Target "cmTC_dfc1b" requires the language dialect "CUDA17" (with compiler extensions), 
but CMake does not know the compile flags to use to enable it.
```
**Solution:** Update CMake to a newer version that supports CUDA17.

