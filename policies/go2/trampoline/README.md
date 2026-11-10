# Go2 trampoline（MJ35）

`config.yaml` + `policy.onnx` 部署 mjlab 的单网络蹦床指令链：站立、蓄能起跳、九种跳跃/空翻、缓冲、站稳。

- 模型来源：`/home/rkz/code/unitree_rl_mjlab-trampoline/logs/remote/MJ35/model_12000.pt`。
- 训练仓库：`Renkunzhao/unitree_rl_mjlab`，分支 `go2-trampoline`；MJ35 训练提交 `7f3bc2d5`，配方后来合入默认配置 `974d209`。
- ONNX 包含经验归一化 `(obs - mean) / (std + 0.01)` 和确定性 actor；不需要额外权重文件。
- 检查点 SHA256：`2644fa640a14477184f44c789feb7532c4aab09a9745bccad6900cdae76ea3ff`。

## 启动

以下命令在 `/home/rkz/code/unitree_ws` 下运行。新增了 `trampoline_command` 观测项，需要先重新编译：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
colcon build --packages-select legged_rl_deploy --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

仿真终端（连接手柄，默认 `/dev/input/js0`）：

```bash
source src/unitree_lowlevel/scripts/setup.sh lo jazzy
ros2 run unitree_mujoco unitree_mujoco \
  --config "$PWD/src/unitree_mujoco/simulate/config/go2_trampoline.yaml"
```

另开终端启动策略：

```bash 
cd /home/rkz/code/unitree_ws
./src/legged_rl_deploy/scripts/run.sh lo jazzy \
  ros2 run legged_rl_deploy legged_rl_deploy_node lo \
  src/legged_rl_deploy/policies/go2/trampoline/config.yaml
```

确认程序提示后，`L2+A` 进入 FixStand；松开组合键，站稳后按 `START` 进入策略。策略初始指令为站立。
真机将两处 `lo` 换成实际网卡，并按仓库主 README 完成低层控制器准备。

## 手柄指令

| 按键 | 指令 |
|---|---|
| A | 原地跳 / 空翻后的缓冲跳 |
| B | 收跳并站稳 |
| 方向键 ↑ / ↓ / → / ← | 前 / 后 / 右 / 左翻一周 |
| 按住 L1，再按方向键 | 对应方向翻两周 |
| SELECT | 返回低层 FixStand |
| L2+B | 低层急停 |

指令在按下时更新，松开后保持；L2 组合键不发送蹦床技能指令。
按训练中的衔接顺序使用：**站立 → A 起跳 → 空翻 → A 缓冲跳 → B 站稳**。
在起跳后、下一次触床前提前给下一条指令；站稳至少 2 秒后再起跳。
本配置是人工切换指令，不包含训练评估器的自动触床调度；空翻后需手动切回 A，避免一直重复同一空翻。

`trampoline_command.params.heights` 分别设置原地跳、单周、两周的世界坐标顶点高度（米），默认均为 1.3。
训练范围分别为 `[0.5, 1.3]`、`[0.7, 1.3]`、`[1.0, 1.3]`；超范围配置会拒绝加载。

## 接口与验证

- 50 Hz 策略、200 Hz 低层；关节按 FL/FR/RL/RR，每腿 hip/thigh/calf，配置映射到硬件顺序。
- 输入 `[1,1046]`：command(10)、gravity(3)、angular velocity(3)、relative q(12)、dq(12)、last action(12)，**每项各自 20 帧、从旧到新**，最后附加 6 个零。首次历史重复首帧。
- 输出 `[1,12]` 为原始动作；PD 目标为 `action * scale + default_pose`，不裁剪目标；只有 last action 观测裁剪到 ±100。
- 编译通过；64 组输入的 ONNX/PyTorch 最大误差 `2.98e-6`。实际 C++ 部署器 32 帧历史/关节映射与参考最大误差 `1.19e-7`，与 Python ORT 动作输出一致；手柄编码、短按保持、重置和非法高度检查通过。
- 尚未验证 ROS/MuJoCo 闭环表现或真机效果；现有仿真场景的机器人、床面和求解器参数与训练环境并非完全一致。

仓库当前忽略 `*.onnx`；模型已放在本目录，复制部署时请同时带上它。
