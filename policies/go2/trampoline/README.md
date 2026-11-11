# Go2 trampoline

`config.yaml` 部署 mjlab 的单网络蹦床指令链：站立、蓄能起跳、九种跳跃/空翻、缓冲、站稳。
当前硬件候选权重是 `policy_chain_deploy_2ms_s42_11250.onnx`（2 ms 对齐）。

> **重要区分（2026-11-11）**：`policy.onnx` 是 MJ35 **5 ms** 模型（15.206408 kg 训练资产、
> native position actuator）。把它直接放进 **2 ms / 16.514412 kg / PGS / explicit PD** 的部署
> 场景，站不住是预期，主要来自训练/部署仿真不匹配，不是单纯初始站姿训练缺口。2 ms 对齐模型是
> `policy_chain_deploy_2ms_s42_11250.onnx`，`START` 后直接由策略接管也能站住。录失败对比时用
> `config_chain_deploy_2ms_mj35_nopatch.yaml`（旧 MJ35）作为失败例。

## 部署基线：模型质量与物理步长

- **推荐基线**：`scene_trampoline.xml` + `go2.xml`，固定 `timestep=0.002`、PGS、`implicitfast`。
- `go2.xml` 的源码记录为 `<!-- total mass: 16.514412 kg, hw: 16.45 kg -->`
  （`unitree_mujoco/unitree_robots/go2/go2.xml:67`；该行随 rotor/foot 质量提交
  `fba3050` 于 2026-09-02 加入）。**16.45 kg 是硬件记录，16.514412 kg 是该修订的
  模型显式质量**，两者相差 0.064412 kg（约硬件的 0.392%）。仓库没有称重照片、
  日期、秤精度、电池/负载清单，因此只能表述为“源码记载的硬件重量”。
- 训练资产的诊断场景（`scene_trampoline_training.xml` / `go2_training_trampoline.xml`，
  显式体质量求和 **15.206408 kg**）已于 2026-11-11 随 implicit-PD 实验一起移除。
  它曾用于说明训练资产比硬件记录轻约 1.244 kg（7.56%）：那是“训练资产对齐”而不是
  “硬件标定”，不能作为质量基线。
- 训练环境本身使用 5 ms/Newton（`unitree_rl_mjlab-trampoline` 的 `env_cfg.py`）；
  那是训练时钟的历史值，不是部署建议。部署统一为 2 ms。
- ONNX 未改动；`policy.onnx` SHA256 为
  `d1d0ba2baae1b48dbbc057e46d0d77bc41f54c2539bba86809213efc9387a8f7`。
  检查点 `model_12000.pt` 的 SHA256 是
  `2644fa640a14477184f44c789feb7532c4aab09a9745bccad6900cdae76ea3ff`（两者不是同一文件）。
- `ll_dt: 0.005` 是低层命令周期，不是 MuJoCo 物理步长；物理步长由场景
  `timestep` 决定，当前为 0.002。

模型来源与导出：

- 模型来源：`/home/rkz/code/unitree_rl_mjlab-trampoline/logs/remote/MJ35/model_12000.pt`。
- 训练仓库：`Renkunzhao/unitree_rl_mjlab`，分支 `go2-trampoline`；MJ35 训练提交 `7f3bc2d5`，配方后来合入默认配置 `974d209`。
- ONNX 包含经验归一化 `(obs - mean) / (std + 0.01)` 和确定性 actor；不需要额外权重文件。ONNX 未因本次 2 ms 对照而改动。

## 启动

以下命令在 `/home/rkz/code/unitree_ws` 下运行。新增了 `trampoline_command` 观测项，需要先重新编译：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
colcon build --packages-select unitree_lowlevel unitree_mujoco legged_rl_deploy --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

仿真终端（连接手柄，默认 `/dev/input/js0`；**默认使用 2 ms 的 go2.xml 基线**）：

```bash
source src/unitree_lowlevel/scripts/setup.sh lo jazzy
ros2 run unitree_mujoco unitree_mujoco \
  --config "$PWD/src/unitree_mujoco/simulate/config/go2.yaml" \
  --eval "$PWD/src/unitree_mujoco/simulate/config/eval_trampoline.yaml"
```

另开终端启动策略：

```bash 
cd /home/rkz/code/unitree_ws
./src/legged_rl_deploy/scripts/run.sh lo jazzy \
  ros2 run legged_rl_deploy legged_rl_deploy_node lo \
  src/legged_rl_deploy/policies/go2/trampoline/config.yaml
```

确认程序提示后，`L2+A` 进入 FixStand；松开组合键，站稳后按 `START`，策略立即接管并负责保持站姿。
部署端**没有**待命零动作阶段：`hold_stand_until_hop` 及其两个状态位已删除。初始站姿在训练侧
用 `initial_stand` 让策略在激活状态下学习；`START` 后若姿态下沉，应检查策略与场景是否对齐，
而不是在部署端加补丁。
曾试验「待命时运行站姿策略以填充真实 `last_action_`/history」，2 ms 实测（2026-11-11）
把首个 hop 顶点压到约 0.95 m、entry hop 普遍失败，该实验已放弃。

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
本配置是人工切换指令；空翻后需手动切回 A，避免一直重复同一空翻。
自动复现、逐技能测试和完整缓冲/站稳流程见 [自动 sim2sim](eval/sim2sim_trampoline_evaluation.md)。

`trampoline_command.params.heights` 分别设置原地跳、单周、两周的世界坐标顶点高度（米），默认均为 1.3。
训练范围分别为 `[0.5, 1.3]`、`[0.7, 1.3]`、`[1.0, 1.3]`；超范围配置会拒绝加载。

## 接口与验证

- 50 Hz 策略、200 Hz 低层；关节按 FL/FR/RL/RR，每腿 hip/thigh/calf，配置映射到硬件顺序。
- 输入 `[1,1046]`：command(10)、gravity(3)、angular velocity(3)、relative q(12)、dq(12)、last action(12)，**每项各自 20 帧、从旧到新**，最后附加 6 个零。首次历史重复首帧。
- 输出 `[1,12]` 为原始动作；PD 目标为 `action * scale + default_pose`，不裁剪目标；只有 last action 观测裁剪到 ±100。
- 编译通过；64 组输入的 ONNX/PyTorch 最大误差 `2.98e-6`。实际 C++ 部署器 32 帧历史/关节映射与参考最大误差 `1.19e-7`，与 Python ORT 动作输出一致；手柄编码、短按保持、重置和非法高度检查通过。
- ROS/MuJoCo 闭环调试结果及复现命令见上述自动 sim2sim 文档。当前推荐场景是
  `scene_trampoline.xml`（`go2.xml`，2 ms，显式 motor PD）；历史 5 ms 训练资产/
  隐式 PD 数字仅作对照，不再作为部署基线。

仓库当前忽略 `*.onnx`；模型已放在本目录，复制部署时请同时带上它。
