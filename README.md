# Legged RL Deploy

This project is for deploying reinforcement learning (RL) policies to Unitree Go2 / G1 robots.

## Features
- Configurable via `config.yaml`
- Supports inference with TorchScript and ONNX Runtime
- Includes several pretrained policies

### Pretrained policy sources
| Source | Policies |
|---|---|
| [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | `go2_velocity_unitree` |
| [My_unitree_go2_gym](https://github.com/yusongmin1/My_unitree_go2_gym) | `go2_jump`, `go2_handstand`, `go2_legstand`, `go2_spring_jump`, `go2_trot` |
| [mjlab](https://github.com/mujocolab/mjlab) | `go2_velocity_mjlab` |

Thanks to the authors of these projects.

## Video results

- [Go2 hop (Bilibili)](https://www.bilibili.com/video/BV1FTzfBcEL6/?vd_source=a178b41776a8e28bb0ca67b41f8f1fe8#reply287915981505)
