# Legged RL Deploy

A deployment framework for reinforcement learning (RL) policies on Unitree Go2 and G1 robots.

## Features

- Configurable via `config.yaml`
- Support for TorchScript and ONNX Runtime inference
- Multiple pretrained policies included

### Policy sources

```
policies/
├── go2/
│   ├── velocity/
│   │   ├── unitree_rl_lab/    # unitree_rl_lab
│   │   ├── mjlab/             # mjlab
│   │   └── aba/               # mjlab
│   ├── hop/                   # My_unitree_go2_gym
│   ├── trot/                  # My_unitree_go2_gym
│   ├── handstand/             # My_unitree_go2_gym
│   ├── legstand/              # My_unitree_go2_gym
│   └── spring_jump/           # My_unitree_go2_gym
└── g1/
    ├── velocity/
    │   └── unitree_rl_mjlab/  # unitree_rl_mjlab
    └── mimic/
        ├── gangnam_style/     # unitree_rl_lab
        ├── dance_102/         # unitree_rl_lab
        ├── dance1_subject2/   # unitree_rl_mjlab
        └── pose/              # whole_body_tracking
        └── LAFAN1_Retargeting # whole_body_tracking
```

| Source | Link |
|--------|------|
| unitree_rl_lab | [github.com/unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) |
| My_unitree_go2_gym | [github.com/yusongmin1/My_unitree_go2_gym](https://github.com/yusongmin1/My_unitree_go2_gym) |
| mjlab | [github.com/mujocolab/mjlab](https://github.com/mujocolab/mjlab) |
| unitree_rl_mjlab | [github.com/mujocolab/unitree_rl_mjlab](https://github.com/mujocolab/unitree_rl_mjlab) |

*Thanks to the authors of these projects for their contributions.*

## Video Demonstrations

- [Go2 Hop (Bilibili)](https://www.bilibili.com/video/BV1Jh6eBhEYz/?share_source=copy_web&vd_source=b99eccd82d555461fbd654f2947e809b)

## Installation

### Dependencies

- [LibTorch](LibTorch.md)

- [ORT](scripts/get_ort.sh)

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

### Run
```bash
# Run test and stop the default controller (run once per boot)
source src/unitree_lowlevel/scripts/setup.sh <network-interface> $ROS_DISTRO
./build/unitree_sdk2/bin/go2_stand_example $NetworkInterface

source src/unitree_lowlevel/scripts/setup.sh <network-interface> $ROS_DISTRO
ros2 run legged_rl_deploy legged_rl_deploy_node $NetworkInterface $WORKSPACE/src/legged_rl_deploy/config/go2-trot.yaml
```

### TeleOP
```bash
sudo apt install -y libhiredis-dev
```

#### Troubleshooting

**CMake CUDA Dialect Error:**
```
Target "cmTC_dfc1b" requires the language dialect "CUDA17" (with compiler extensions), 
but CMake does not know the compile flags to use to enable it.
```
**Solution:** Update CMake to a newer version that supports CUDA17.

