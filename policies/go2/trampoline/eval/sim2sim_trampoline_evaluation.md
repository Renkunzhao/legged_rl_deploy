# Trampoline 自动 sim2sim

使用真实 `unitree_mujoco`、ROS 低层控制器和 C++ `PolicySlot`，通过
`/wirelesscontroller` 模拟按键。复用 beam 流程的控制器握手、reset/start 服务，
蹦床评价不使用 beam 的倾倒终止条件。

## 编译与运行

```bash
cd /home/rkz/code/unitree_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
colcon build --packages-select unitree_lowlevel unitree_mujoco legged_rl_deploy \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

ROS_DOMAIN_ID=83 bash src/legged_rl_deploy/policies/go2/trampoline/eval/run_trampoline_sim2sim.sh \
  --output results/trampoline/run01 \
  --sim-config src/unitree_mujoco/simulate/config/go2.yaml \
  --sim-eval src/unitree_mujoco/simulate/config/eval_trampoline.yaml \
  -- --trials 3 --full-chain --require-energy-height
```

`--output` 必须是新目录。启动器绑定 loopback，禁用实体手柄，启动仿真及
`evaluation_mode:=true` 控制器，结束时清理自己启动的进程。使用一个空闲的非零 ROS domain。
`--sim-config`、`--sim-eval` 和 `--policy-config`（放在 `--` 前）可指定对照配置。

**推荐基线**是 `--config config/go2.yaml --eval config/eval_trampoline.yaml` → `scene_trampoline.xml` + `go2.xml`：
2 ms、PGS、`implicitfast`，机器人模型显式质量 16.514412 kg（源码记录的硬件重量
16.45 kg），显式 `<motor>` PD。`ll_dt: 0.005` 只是低层命令周期，不是物理步长。

### 已移除的诊断：执行器力矩旁路与 implicit PD

2026-11-09 曾做过一组 2 ms 配对诊断：低层用 `simulator_actuator_torque_limits:=true`
跳过 `torqueClip()` 的延迟状态预限矩，并对比 `go2.xml` 的显式 `<motor>` PD 与训练资产的
affine-bias implicit PD。结论是显式 PD 在 hop/entry 能量与完整链上更好，implicit 的 hop
顶点稳定低于 1.20 m 蓄能门限。

该实验依赖的代码、配置和训练资产场景已于 2026-11-11 全部移除，因此启动器不再提供
`--actuator-torque-limits`。原始结果保留在 `results/trampoline/fixed2ms_ablation/`，
仅作历史记录；当前基线是 `go2.xml` 的显式 motor PD + 低层默认 `torqueClip()`。

已有常驻栈时也可单独运行：

```bash
python3 src/legged_rl_deploy/policies/go2/trampoline/eval/eval_trampoline_sim2sim.py \
  --output results/trampoline/resident01 --skills stand front-1 back-1 front-2 \
  --full-chain
```

仿真 YAML 需要 `episode_services: true`、`eval: ""`，启动时传
`--no-joystick`。环境要求同 beam：显式非零 `ROS_DOMAIN_ID`、
`NetworkInterface=lo`、CycloneDDS 显式绑定 `lo`。不依赖深度相机。
控制器用 `--ros-args -p evaluation_mode:=true`。

## 流程与数据

每条技能独立 reset：FixStand → START → 站姿观测 → A 起跳 → 指定技能。
`--full-chain` 追加 A 缓冲跳 → B → 站稳观测。

- 技能：`stand hop front-1 back-1 right-1 left-1 front-2 back-2 right-2 left-2`。
- 默认在顶点发送下一条命令；`--switch-phase takeoff` 在首个离地样本后 20 ms 发送
  （对齐训练起跳时 reveal 的时序），`--switch-phase landing` 对照触床时发送。
- `--require-energy-height` 在首次 hop 达到目标 ±0.1 m 后才切空翻；
  否则继续 hop。用于对齐训练的蓄能判据；复现手动直接切换时不启用。
- `--height` 仅设评价目标，须与 policy YAML 的 heights 对齐，不修改策略指令。
- `--stand-time` 默认 3 秒；`--skill-timeout` 默认 10 秒。

启动器保存模型 SHA256、仓库版本/改动状态与两份 YAML 快照。
输出 `summary.json`、`results.json`、每条技能的 `*.trace.jsonl`，包括仿真时间、
世界位置/速度、IMU、关节状态、接触力、实际发送按键、PD 目标及有效力矩。
flight 记录顶点、累计旋转（含起跳前倾角）、落地倾角、偏航差及位移。
`rotation_landing_ok` 与 `height_ok` 分开记录；`success` 要求两者均通过。
初始站立及落地站稳分别记录，不能将“翻完但跳低了”计为严格通过。
`chain_success` 还要求进入目标技能前的最后一次 hop、缓冲 hop、初始待命和最终站姿均合格。
站姿使用最后 2 秒的全部关节偏差（hip/thigh/calf 0.2/0.3/0.3 rad）、
左右镜像误差 0.2 rad、四脚承载及速度/稳定性检查。
`body_height` 是离脚底高度；评分用 `body_height_center`（减去 0.022 m 脚半径），门限 0.2 m。
基础设施错误用 `error_phase` 单独记录，进程退出码非零；技能未达标保留完整结果供分析。

这是基于 ROS 状态的部署诊断器：触床由脚力滞回判定，状态流异步；没有训练环境的
逐物理子步自碰撞检测。因此其通过率不等同训练评估的 256-world survival rate。

## START 后的站姿

部署端没有待命零动作阶段。`START` 后策略立即接管并负责保持站姿；第一次 `A` 开始起跳，
`B` 是网络的收跳站稳命令，`SELECT` 回到 FixStand。

`hold_stand_until_hop` 及其两个状态位已于 2026-11-11 删除。它原本把 MJ35 训练中只持续
0.3–1 秒的 reset stand 延长成操作员等待窗口，是部署侧补丁；永久方案在训练侧，用
`initial_stand` 让策略在激活状态下学会初始站姿。曾试验「待命时运行站姿策略以积累真实
history」，2 ms 实测把首个 hop 顶点压到约 0.95 m、entry hop 普遍失败，该实验已放弃。

`START` 后若姿态下沉，应优先检查策略与场景（质量/步长/PD）是否对齐，而不是在部署端
重新加待命分支。

## 历史调试记录（2026-11-09，非当前基线）

> **这些结果不是当前 2 ms / `go2.xml` 基线。** 它们使用培训资产或 5 ms 场景、
> 不同的执行器表达/低层预限矩配置，只用于说明当时的排查过程。当前推荐基线的
> 配对结果见 `results/trampoline/fixed2ms_ablation/` 与下节更新。

本机结果保存在 workspace 的 `results/trampoline/`：

| 对照 | 结果 |
|---|---|
| `baseline02`：原场景、原 START 行为 | START 后机身离脚约 0.086 m；hop 顶点约 1.13 m；前双周约 0.96 m 且欠转 |
| `standby_long`：仅启动待命站姿 | 两次 10 秒保持，机身离脚约 0.255 m，关节速度 RMS 约 0.002 rad/s |
| `newton2_legacy`：仅换 Newton、仍 2 ms | hop 约 1.12 m，仍偏低 |
| `newton5_legacy02`：旧机器人、Newton / 5 ms | hop 1.31 m、前单周通过；前双周仍欠转 |
| `training_asset01`：匹配机器人、显式 PD | 站立关节速度 RMS 约 3–6 rad/s，部分 reset 失稳 |
| `implicit01`：再改为隐式 PD、仍低层预限矩 | 抖动消失，待命约 0.005 rad/s；hop 约 1.0 m，蓄能门限阻止进入空翻 |
| `actuator_saturation_only`：仿真执行器限矩 | hop 1.27 m；前双周 1.22 m、约 12.96 rad、落地倾角 0.40 rad，通过该次测试 |

最后一项通过临时配置隔离低层预限矩的影响；该临时开关已于 2026-11-11 移除。
早期记录的旋转角未全部使用离床前的固定航向参考，因此不能直接用早期角度值统计新版通过率。

### 历史“修复后”闭环结果（5 ms 训练资产 + implicit，非当前基线）

`final_3x/evaluation` 每项 3 次，加上最终启动器完整重跑
`verified01/evaluation` 每项 1 次。**这些运行使用 `scene_trampoline_training.xml`
的 5 ms/Newton 训练资产 + `implicit_pd: true`，不是当前 2 ms `go2.xml` 基线，
不能作为当前通过率声明；相关诊断资产已于 2026-11-11 移除，记录仅作历史。**
固定床质量 10 kg、MJ35 ONNX：

| 指令 | 目标技能通过 | 完整链通过 | 目标顶点范围 / m |
|---|---:|---:|---:|
| hop | 4/4 | 4/4 | 1.274 |
| 前单周 | 4/4 | 4/4 | 1.293–1.326 |
| 后单周 | 4/4 | 4/4 | 1.293–1.344 |
| 右单周 | 4/4 | 3/4 | 1.278–1.327 |
| 左单周 | 4/4 | 4/4 | 1.295–1.301 |
| 前双周 | 4/4 | 4/4 | 1.218–1.221 |
| 后双周 | 4/4 | 4/4 | 1.182–1.189 |
| 右双周 | 4/4 | 3/4 | 1.160–1.169 |
| 左双周 | 4/4 | 4/4 | 1.216–1.231 |

八种空翻目标共 **32/32**；含 hop 的完整链 **34/36**。两次完整链失败均是
右翻之后的缓冲 hop 偏航超过 30°，目标空翻和最终站稳均通过，未放宽判据。
`final_stand10s/evaluation` 两次 10 秒待命站姿均通过，机身高出脚中心约
0.238 m，关节速度 RMS 约 0.0021 rad/s。
`tau_est` 峰值为 23.700001 / 45.430000 Nm（ROS float 精度），实际执行器限矩保持有效。

汇总：`results/trampoline/validation_summary.json`。`final_3x` 的 30 个评估均完成；
运行期间启动脚本被编辑，导致评估结束后 shell 解析退出码 2，原始结果完整保留。
随后固定脚本、使用最终版本重新跑完 `verified01` 的全部 10 项，退出码 0。
最终汇总还重新检查了进入技能的 hop 和启动站姿要求。

## 当前推荐（2026-11-09）

1. 仿真/部署基线用 `scene_trampoline.xml` + `go2.xml`（16.514412 kg 模型，
   源码记录 hw 16.45 kg），固定 2 ms、PGS、implicitfast。不要为单个 checkpoint
   换成 15.206408 kg 训练资产或 5 ms 时钟。
2. PD 用 `go2.xml` 原始 `<motor>` 的显式 PD；implicit affine 诊断路径已随实验移除。
3. 低层默认使用自己的 `torqueClip()`；loopback 旁路已移除，仿真与真机走同一套
   限幅语义。
4. `front-2` 与完整链残余应通过面向 16.514412 kg、2 ms 的训练适配解决；不要修改
   质量、求解器、步长、力矩上限或增益来制造通过。训练端若保持 20 ms 策略周期，
   2 ms 物理步长应使用 decimation 10。

验证命令：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 -m pytest src/legged_rl_deploy/policies/go2/trampoline/eval/test_trampoline_eval.py -q
ctest --test-dir build/unitree_mujoco --output-on-failure
```

16 项测角/站姿/链路单测和 23 项 MuJoCo 场景/执行器检查通过。本次只改 Python
评测器、XML/YAML 和文档，没有改 C++ 源码，因此未重新编译三个包。
