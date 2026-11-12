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

删除补丁的代价在下一节量化：没有零动作待命，起跳前的姿态会站得更直，hop 顶点从
1.38 m 掉到约 0.88 m。这条补丁**不能靠恢复来修**，必须由训练侧的 `initial_stand`
让策略学会从自持站姿起跳。

`START` 后若姿态下沉，应优先检查策略与场景（质量/步长/PD）是否对齐，而不是在部署端
重新加待命分支。

## 11250 的 hop 顶点为何从 1.38 m 跌到 0.88 m（2026-11-11 定位）

问题：`policy_chain_deploy_2ms_s42_11250.onnx` 曾在 `takeoff_run01` 得到 17/18 目标、
15/18 完整链；同一 SHA256 在今天当前代码上重跑只有 1/18 与 0/18，hop 顶点从
1.38 m 掉到 0.88 m，全部卡在 1.20 m 蓄能门限之下，后续空翻连锁失败。

结论：**不是权重、不是导出、也不是仿真物理退化**，而是删掉 `hold_stand_until_hop`
待命补丁后，`A` 起跳时机的起始姿态变了。三次运行 `manifest.json` 的模型 SHA256 完全
相同（`d6624a57cb11…`），可排除 checkpoint 混用。

| 运行 | 站姿由谁维持 | 稳态 `body_height` | 稳态 thigh (前/后) | hop 顶点 | 目标/链 |
|---|---|---:|---|---:|---|
| `takeoff_run01` | 零动作 + PD 默认姿态（`standby_policy: false` + `hold_stand_until_hop: true`） | **0.259 m** | 0.81 / 0.71 | **1.38 m** | 17/18, 15/18 |
| `takeoff_standby_run01` | 策略在 stand 指令下自持（`standby_policy: true`） | 0.293 m | 0.87 / 1.08 | 0.95 m | 13/18, 0/18 |
| `recheck_run01` | 策略在 stand 指令下自持（补丁已删，`START` 即接管） | 0.293 m | 0.86 / 1.05 | 0.88 m | 1/18, 0/18 |

机制：零动作待命把机器人压进更深的屈腿姿态（0.259 m），起跳时腿有更大的伸展行程，
因此蓄能更高。策略自持站姿把它站得更直（0.293 m），行程被吃掉，hop 约 0.9 m。
`takeoff_standby_run01` 与 `recheck_run01` 走两种实现但落到同一个姿态与同一个顶点，
说明主导变量是**起跳前的姿态**，不是待命代码路径本身。

分开的时刻可以精确定位：两次运行在 `START` 之前逐点一致（0–3.2 s 的 `body_height`
差 < 3 mm，FixStand 后的共同抬升都到 0.322 m），从 3.25 s 起分叉——正是策略接管
站姿的那一刻。之后到 6 s 稳定在 0.259 m 与 0.293 m 两条水平线。物理模型、`go2.xml`、
`trampoline.xml`、`scene_trampoline.xml`、`ros2_bridge.cc` 在两个 commit 间均逐字节
未变，因此这不是场景或积分器差异。

同时排除了此前怀疑的 `torqueClip()` 旁路：`takeoff_standby_run01` 带旁路（`bypass=True`）、
`recheck_run01` 不带旁路，两者顶点只差 0.95 → 0.88 m（约 7 cm），是次要项。
全量 trace 统计中执行器到限的时间占比也只有 0.05–0.44%（逐时间步 0.30–2.91%），
限幅本身不是瓶颈。

因此：**0.984 是训练侧 128-world `strict` 生存率**（`eval/nominal_strict_model_11999.json`，
`mean_seconds` 19.96 / 20），与部署侧 sim2sim 通过率不是同一个量。训练侧给策略的初始
条件是 `initial_stand`，策略在激活状态下从一开始就自持站姿并学会从该姿态起跳；
ppo01 快照里是 `hold_in_wait=deployment_aligned` + `CommandChainCfg()` 默认
`initial_stand=False`，即**训练时有零动作待命、部署时才删掉**，两边不一致。
ppo02b 快照已改为 `CommandChainCfg(initial_stand=deployment_aligned)`，是这条链路上
真正需要验证的改动。

结论优先级：`initial_stand` 是否真的让策略从自持站姿起跳达到 1.3 m，决定 11250 能否
上机；在此之前不要恢复待命补丁，也不要调质量/步长/求解器/力矩上限。

## 手动命令与评测环境的一致性（2026-11-11 核对）

对照用户的两条手动命令逐项核对：

| 项 | 手动 | 自动评测 | 是否一致 |
|---|---|---|---|
| 场景 | `--eval simulate/config/eval_trampoline.yaml` | 同一文件 | ✅ 一致 |
| 物理步长 | 2 ms（`go2.xml`） | 同一 `go2.xml` | ✅ 一致 |
| 低层 | `src/unitree_lowlevel/config/go2.yaml` | 同一文件 | ✅ 一致 |
| 策略 | `config.yaml` → `..._11250.onnx` | `config_ab_11250.yaml` → 同一 SHA256 | ✅ 一致 |
| `evaluation_mode` | **未启用** | `:=true` | ❌ 见下 |
| 手柄来源 | 实体 `/dev/input/js0` | 合成 `/wirelesscontroller` | ❌ 见下 |
| 观众窗口 | 开（`no_viewer: false`） | `--no-viewer` | ⚠️ 见下 |
| 下一指令时机 | 人眼判断 | `velocity_z <= 0` 且离地 >0.1 s | ❌ 见下 |
| 按键保持 | 人手 | 固定 60 ms | ⚠️ 见下 |
| 指令到期后的链后续 | 手动决定 | `--full-chain` 自动 | ❌ 见下 |
| 判定 | 人眼"站住了/摔了" | 旋转 ±45° + 偏航 ±30° + 倾角 ≤30° + 位移 <0.75 m + 高度 ±0.1/0.2 m | ❌ 见下 |

**`simulator_actuator_torque_limits:=true` 已经是空操作。** 该参数连同它依赖的代码、
配置和训练资产已于 2026-11-11 全部移除：源码 0 处、构建产物 0 处、`strings` 里 0 次。
ROS 2 对未声明的 `--ros-args -p` 只告警不报错，所以手动命令照常启动，PD 语义与评测
一致。可以放心删掉这个参数。

**用户命令里的相对路径只在 `src/unitree_mujoco` 下成立。** `simulate/config/go2.yaml`
相对 `unitree_ws` 根不存在（相对 `unitree_ws/src/unitree_mujoco` 才存在）。用户当时
的 sim 进程 `cwd` 正是 `src/unitree_mujoco`，所以能跑起来；从 `unitree_ws` 根启动会
`bad file`。这不是行为差异，只是路径陷阱。

### 手动与评测的行为差异（按影响排序）

1. **下一指令的时机。** 评测器在 `velocity_z <= 0 且 t-takeoff > 0.1 s` 的第一个样本
   发指令，即顶点后 4–6 ms。人眼判断顶点有 100–250 ms 量级的延迟和抖动（见下节）。
2. **`evaluation_mode` 关闭**时没有 `/legged_rl_deploy/status`、没有 reset/start 服务、
   没有 `policy_reset_sequence` 握手，csv 日志也不会在每个 episode 前清空。手动跑不会
   因为状态服务缺席而失败，但也没有可回放的 episode 边界。
3. **观众窗口开着。** physics 线程按实时比例节流（`percentRealTime` 目标 100%），
   评测是无窗口全速。这改变墙钟，不改变仿真时间语义——所以不影响物理，只影响
   人能按键的密度。
4. **L1 必须与方向键同一拍**：`skill = gamepad.L1.pressed ? 3 : 2`，`L1.pressed` 在方向键
   `on_press` 那一拍读取。先按方向键再补 L1 不会变成双翻，会按单翻执行。手感上很
   容易漏。
5. **判定口径不同。** 手动"看起来成了"通常只要求落地后不摔；评测还要求旋转落在
   ±45°、偏航 ±30°、倾角 ≤30°、位移 <0.75 m、以及顶点高度达到 1.3 m ±0.1（单翻）
   / ±0.2（双翻）。当前策略**全部**卡在高度这一项上（见下），所以无论手动还是自动，
   严格的 `success` 都是 0。

### 当前 11250 的真实通过口径（`apex_ab_11250_n5`，45 目标飞行）

| 口径 | 单翻 (n=20) | 双翻 (n=20) |
|---|---:|---:|
| 严格 `success`（旋转 + 高度） | 1 (5%) | 0 (0%) |
| `rotation_landing_ok` (±45°) | 18 (90%) | 11 (55%) |
| 落地倾角 ≤30° | 18 (90%) | 11 (55%) |
| 落地倾角 ≤60°（"没摔"） | 20 (100%) | 18 (90%) |
| 顶点高度合格 | 1/20 | 0/20 |

**结论：现在与翻转质量无关的失败项是顶点高度。** hop 顶点只有 0.85–0.89 m，空翻顶点
1.01–1.16 m，而门限是 1.30 m。这与删除 `hold_stand_until_hop` 造成的起跳姿态变化是
同一件事（上一节），不是新问题。所以"我测出来成功率高得多"这句需要先对齐口径：
若按"落地没摔"算，自动评测的单翻是 100%、双翻是 90%。

### `apex_ab_*`：复现手动 bug 的 A/B 评测流程（2026-11-12）

目的：在**同一份部署代码、同一物理、同一按键时序**下，对 11250 与 11999 做 5-trial
全流程评测，确认手动侧"双翻转不过来、落地歪"不是操作问题，而是当前策略在部署环境里的
真实表现；同时留下一条与手动 bag 同口径的可复现基线。

#### 启动方式

两份 policy YAML 只差 `model_path`，观测项、动作 scale/offset、PD 增益、
`policy_dt: 0.02`、`ll_dt: 0.005` 完全相同：

- `policies/go2/trampoline/config_ab_11250.yaml` → `policy_chain_deploy_2ms_s42_11250.onnx`
- `policies/go2/trampoline/config_ab_11999.yaml` → `policy_chain_deploy_2ms_s42_11999.onnx`

仿真侧用默认机器人 profile `src/unitree_mujoco/simulate/config/go2.yaml` 加任务 overlay
`src/unitree_mujoco/simulate/config/eval_trampoline.yaml`：2 ms、16.514412 kg、PGS /
`implicitfast`、显式电机 PD，未改质量、惯量、求解器、时间步或力矩上限。run 目录里的
`simulator.yaml` 是机器人 profile（`scene.xml`），`simulator_eval.yaml` 才是任务
overlay（`scene_trampoline.xml` + `episode_services`），实际生效的是两者合并后的蹦床场景。

两个 run 用不同 ROS domain 并行：

```bash
cd /home/rkz/code/unitree_ws
source /opt/ros/jazzy/setup.bash && source install/setup.bash

ROS_DOMAIN_ID=101 bash src/legged_rl_deploy/policies/go2/trampoline/eval/run_trampoline_sim2sim.sh \
  --output results/trampoline/apex_ab_11250_n5 \
  --policy-config src/legged_rl_deploy/policies/go2/trampoline/config_ab_11250.yaml \
  -- --trials 5 --full-chain --switch-phase apex

ROS_DOMAIN_ID=102 bash src/legged_rl_deploy/policies/go2/trampoline/eval/run_trampoline_sim2sim.sh \
  --output results/trampoline/apex_ab_11999_n5 \
  --policy-config src/legged_rl_deploy/policies/go2/trampoline/config_ab_11999.yaml \
  -- --trials 5 --full-chain --switch-phase apex
```

评测参数固定为 `--trials 5 --full-chain --switch-phase apex`，评分高度 `--height 1.3`，
`require_energy_height` 保持 `false`（与 11250 的旧 run 相同），因此结果差异不来自评分
开关。后台启动仍需 `setsid ... < /dev/null & disown`，否则控制器会在启动器清理阶段收到
`rcl_shutdown already called`。

#### 模型与代码版本

| 项 | 11250 arm | 11999 arm |
|---|---|---|
| ONNX | `policy_chain_deploy_2ms_s42_11250.onnx` | `policy_chain_deploy_2ms_s42_11999.onnx` |
| sha256 | `d6624a57cb113dbc99850313701c99797535da75a1776ffe24e59d688c6dce85` | `3fadd1de4287b2bd48a7351a7841ff86602f36abfca254c7cf260cfe560949d2` |
| `legged_rl_deploy` HEAD | `fca11b07f8` | `fca11b07f8` |
| `unitree_lowlevel` HEAD | `0fd9213943` | `0fd9213943` |
| `unitree_mujoco` HEAD | `a978ac326d` | `a978ac326d` |

两个 run 的 `manifest.json` 记录的三仓库 HEAD 相同；当时 `legged_rl_deploy` 工作区只有
这两份 A/B YAML 未跟踪。11999 是 ppo01 的最后一次保存（`max_iterations=12000`），不是
从 11250 续训的中间点。

#### 结果（目标飞行，每技能 n=5）

`rot_land` 为 `rotation_landing_ok`（±45°）；严格成功 = 旋转 + 顶点高度 1.3 m ±0.1/±0.2。

| 技能 | 11250 `rot_land` | 11250 严格成功 | 11250 顶点 | 11999 `rot_land` | 11999 严格成功 | 11999 顶点 |
|---|---:|---:|---:|---:|---:|---:|
| front-1 | 5/5 | 0/5 | 1.124 m | 5/5 | 0/5 | 1.167 m |
| back-1 | 3/5 | 0/5 | 1.139 m | 2/5 | 0/5 | 1.128 m |
| right-1 | 5/5 | 1/5 | 1.166 m | 5/5 | 0/5 | 1.124 m |
| left-1 | 5/5 | 0/5 | 1.167 m | 5/5 | 0/5 | 1.148 m |
| front-2 | 1/5 | 0/5 | 0.964 m | 2/5 | 0/5 | 0.945 m |
| back-2 | 0/5 | 0/5 | 0.990 m | 0/5 | 0/5 | 1.034 m |
| right-2 | 5/5 | 0/5 | 1.036 m | 5/5 | 0/5 | 1.049 m |
| left-2 | 5/5 | 0/5 | 1.075 m | 5/5 | 0/5 | 1.079 m |
| hop | 5/5 | 0/5 | 0.869 m | 5/5 | 0/5 | 0.851 m |

所有 run 的稳态站姿高度：11250 为 0.293 m，11999 为 0.285 m；离地瞬间的 hop 顶点只有
0.84–0.86 m。严格成功合计：11250 为 1/45，11999 为 0/45。

判读：

- **单翻基本正常，缺陷集中在俯仰双翻**：`front-2` 1/5、2/5，`back-2` 0/5、0/5，
  而滚转双翻在同一份数据里是 5/5。方向排序与手动一致。
- 11999 相对 11250 没有质变：`front-2` 从 1/5 到 2/5，`back-2` 仍为 0/5，
  `back-1` 从 3/5 退到 2/5。不能把这次 A/B 当作修复。
- 严格成功接近 0 的直接原因是顶点高度（0.85–1.17 m）低于 1.3 m 门限，与翻转质量无关；
  排查翻转时应看 `rotation_landing_ok`，不要用 `success`。

#### 为什么不能和 `chain_deploy_2ms_s42_11250_run01` 的 24/30 横比

| 项 | `chain_deploy_2ms_s42_11250_run01`（旧） | `apex_ab_11250_n5`（本次） |
|---|---|---|
| `hold_stand_until_hop` | `true`（零动作待命补丁） | 已删除，策略自持站姿 |
| 稳态站姿高度 | 0.259 m | 0.293 m |
| entry hop 顶点 | 1.375 m | 0.869 m |
| `front-2` 目标顶点 | 1.136 m | 0.964 m |
| `target_success` | 24/30 | 1/45 |
| `unitree_mujoco` HEAD | `3bdafc5af8` | `a978ac326d` |
| `unitree_lowlevel` HEAD | `453042f8d4` | `0fd9213943` |
| `legged_rl_deploy` HEAD | `6cec3c7880` | `fca11b07f8` |

旧 run 的高成功率来自已删除的 `hold_stand_until_hop`：零动作把机器人压到 0.259 m 的
深屈腿姿态，起跳行程更大，hop 顶点 1.38 m，后续空翻才有多余高度。删掉补丁后策略把
自己站得更直（0.285–0.293 m），腿部伸展行程被吃掉，hop 掉到约 0.87 m，双翻随之没有
余量。中间的 `chain_deploy_2ms_s42_11250_recheck_run01`（补丁已删、2 trials）只有
1/20 目标成功，与本次方向一致。

因此：**`apex_ab_*` 才是与当前代码、当前手动操作同口径的基线；旧 run 的 24/30 不能再
当作现有能力引用。** 修复方向在训练侧（`initial_stand` 与起跳段姿态/角速度约束），
不是恢复待命补丁。

### 方向顺序为什么是"前空翻最差、后空翻其次"

自动评测复现了用户的观察，且机制清楚：

| 技能 | 切换瞬间俯仰角速度 (rad/s) | 计入评分的 `pre_rotation` (rad) | 落地倾角均值 |
|---|---:|---:|---:|
| front-1 | +1.74 ± 0.13 | 0.50 | 12.5° |
| back-1 | +1.62 ± 0.04 | 0.82 | 29.5° |
| right-1 | +0.03 ± 0.17 | 0.22 | 3.3° |
| left-1 | +0.09 ± 0.15 | 0.46 | 6.7° |
| front-2 | +1.73 ± 0.13 | 1.22 | 42.9° |
| back-2 | +1.68 ± 0.08 | 1.50 | 54.4° |
| right-2 | +0.02 ± 0.23 | 1.02 | 19.7° |
| left-2 | +0.09 ± 0.20 | 1.03 | 10.8° |

hop 顶点**不是中性姿态**：机体此时仍以约 1.7 rad/s 向前俯仰。这个俯仰对前后翻的
作用方向相反：

- **前空翻**顺着它转，起跳前已经"借"了 0.50/1.22 rad，但落地时仍是前沿先着地，
  倾角小（12.5°/42.9°）→ 看起来"翻过去了但落地发飘"。
- **后空翻对着它转**，先要把这 1.5–1.7 rad/s 抵消掉，实测 `pre_rotation` 反而累积到
  0.82/1.50 rad，落地倾角大（29.5°/54.4°）→ 看起来"翻不够、要摔"。
- **左右翻**俯仰分量接近 0（0.02–0.09），所以最干净（3.3–10.8°）。

这解释了用户看到的"前空翻最差、后空翻其次"，但真实排序其实是**前后都差、左右好**；
前空翻的"最差"更可能来自落地姿态而非旋转不足。左右翻在同一份自动数据里是 10/10 和
10/10 的 `rotation_landing_ok`。

### 落地判定：deploy 侧 vs 训练侧（逐项对照，2026-11-12）

用户问"deploy 侧的成功落地判定和训练侧是否一致"。逐行比对
`trampoline/eval/eval_trampoline_sim2sim.py` 的 `rotation_landing_ok` /
`height_ok` 与训练 `src/tasks/trampoline/mdp/command_chain.py` 的
`_flight_state()` / `_update_rotation_state()` / `_update_safety_terminations()`：

**阈值完全相同：**

| 量 | 训练 | deploy | 一致 |
|---|---|---|---|
| 落地最大倾角 | `landing_max_tilt` = π/6 | `tilt <= math.pi/6` | ✅ |
| 落地最大偏航误差 | π/6 | `<= math.pi/6` | ✅ |
| 最大水平位移 | 0.75 m | `< .75` | ✅ |
| 转角完成容差 | `rotation_completion_tolerance` = π/4 | `<= math.pi/4` | ✅ |
| 顶点高度容差 | 单 0.1 / 双 0.2 m | 单 0.1 / 双 0.2 m | ✅ |
| 倾角轴 | 机体 z 轴投影 | `R[2,2]` | ✅ |

**不同的是"什么算失败"：**

| 机制 | 训练 | deploy |
|---|---|---|
| 触地倾角 > 45° | `fall_max_tilt` = π/4 → **终止 episode** (`task_failure`) | 不终止，只有 >30° 记 `landing_tilt` |
| 自碰撞 | `self_contact` → 终止 | 无 |
| 物理发散 | `physics_diverged` → 终止 | 无 |
| 飞行平面违规 | `flight_plane_violation`（`min_flight_plane_projection` 0.8）→ 终止 | 无 |
| 起跳超时 | `takeoff_timeout` = 2.0 s → 终止 | 无 |
| 顶点高度 | 是 `rotation_success` 的**与**项 | 独立的 `height_ok`，不影响 `rotation_landing_ok` |
| 触地检测 | 垂直动量残差滤波（30/10，α 0.25） | 足底力求和（10/30 迟滞） |

**关键差异：训练侧把"背部着地"当成终止条件，deploy 侧只把它当成一个评分项。**
所以同一段行为在两个系统里看起来完全不同：训练里 episode 直接结束（记为
`failure/landing_fall`），deploy 里机器人继续躺着、继续被积分。

另一个差异：训练侧 `flight_success = landing_ok & rotation_success`，而
`rotation_success` 还要求 `apex_height_matched`；deploy 侧
`rotation_landing_ok` 不含高度，所以 deploy 会出现"旋转+落地都合格但顶点不够"
的组合，这在训练侧是不可能的。

### 自动评测侧的同一批数据

先更正一个方法学错误：`.trace.jsonl` 按**到达顺序**写入，开头带着上一集的尾部
（sim 时间 8.5→10.9），然后才回绕到 0。跨越回绕点读文件会把上一集的坠地残影
当成这一集，之前得出的"前空翻 32 次背部着地"就是这个假象。正确做法是只保留
最后一个时间下降点之后的那一段。修正后按 `--skills` 口径重新统计：

| 技能 | n | 需求转数 | 实际完成 | 落地倾角均值 | 最大 | `rotation_landing_ok` | 背部着地 |
|---|---:|---:|---|---:|---:|---|---:|
| front-1 | 104 | 1 | 0.98 | 10.8° | 27.1° | 103/104 | 0 |
| back-1 | 44 | 1 | 0.96 | 24.4° | 48.8° | 28/44 | 0 |
| left-1 | 44 | 1 | 1.01 | 8.4° | 20.7° | 44/44 | 0 |
| right-1 | 44 | 1 | 1.00 | 7.9° | 19.1° | 44/44 | 0 |
| **front-2** | 45 | 2 | **1.83** | 33.7° | 172.6° | 20/45 | 4 |
| **back-2** | 45 | 2 | **1.83** | 42.6° | 166.3° | 14/45 | 2 |
| left-2 | 43 | 2 | 1.99 | 12.0° | 26.7° | 43/43 | 0 |
| right-2 | 43 | 2 | 1.98 | 13.8° | 26.4° | 43/43 | 0 |

失败原因分布也指向同一件事（`back-2`：`rotation_angle` 19 次、`landing_tilt` 7 次；
`front-2`：`no_landing_before_timeout` 8 次、`rotation_angle` 7 次）。

### 用户手动日志里"背部着地"的实测（`plot/data/data.csv`）

用户提供 325 s / 102,246 行 / 250 Hz 的手动日志。先做数据体检，避免用错列：

- `real_base_pos_*`、`real_base_lin_vel_W_*`、`real_f_pin_*`、`real_ee3Dof_*`、
  `real_eePos_pin_*`、`real_eeVel_pin_*` **全部为 0 或单值——没有记录**。
  因此高度、世界系速度、足底接触都无法从这份日志得到。
- 可用：`real_base_quat_{0..3}`（**(x,y,z,w)** 顺序）、`real_base_R_22`、
  `real_base_eulerZYX_{0,1,2}`（yaw,pitch,roll）、`real_base_ang_vel_B_{x,y,z}`、
  `real_joint_*`、`jnt_cmd_*`、`state`。
- 每行**稀疏写入**（281 列里只有部分非空），按时间前向填充后才是连续信号。
- `time_s` 严格单调，dt 中位数 4 ms → 250 Hz，无回绕。

姿态来源自校验：四元数算的 tilt 与 `R_22` 算的 tilt，中位差 0.000°，
p99 0.005°，仅 69/102246 个样本超过 2°（翻转最快处）——两路独立一致。

用"体轴角速度积分 + 四元数增量"双路定轴，pitch 轴（body-y）前后翻共 **26 次**：

| 组 | n | 落地直立 (<30°) | 背部着地 (>120°) | 平均滞空 | 平均转速 | 平均峰值 |
|---|---:|---:|---:|---:|---:|---:|
| 单翻尝试 | 7 | 3 | 0 | 0.89 s | 6.4 rad/s | 10.1 |
| 双翻尝试 | 19 | 2 | **10** | 1.01 s | 10.5 rad/s | 13.6 |

按完成转数分桶，失败集中在 1.5 转左右：

| 完成转角 | 事件数 | 落地直立 | 背部着地 |
|---|---:|---:|---:|
| 0.5 转 | 2 | 1 | 0 |
| 1.0 转 | 5 | 2 | 0 |
| **1.5 转** | **16** | **0** | **9** |
| 2.0 转 | 3 | 2 | 0 |

**1.5 转这一簇 16 次全部没站住（0 直立）。** 这不是"转得太快摔过去"，
而是"想转两圈只转了一圈半，差 90–180°，于是以倒立/仰面姿态触地"：
每个事件的 `缺口角 = (2 − 完成转数) × 360°` 与落地倾角的对应关系是
`132°→116°`、`146°→152°`、`169°→172°`、`176°→174°`、`180°→175°`。

左右翻（body-x）在**同一份**日志里是 3 次纯侧翻，全部 2.0 转、落地倾角
3.5–5.9°。所以问题不是"空翻都做不好"，是**前后翻双圈**这一项。

### 为什么双翻转不过来：转速，不是滞空时间

先做了一个容易做错的判断，然后用反事实检验把它推翻了，这里如实记录过程。

**第一版（错的）**：对 26 次前后翻算相关，得到"完成转数 ↔ 峰值转速 = +0.899"、
"完成转数 ↔ 滞空时间 = +0.512"，两次成功的双翻滞空 1.29–1.30 s，17 次失败的只有
0.88–1.21 s，于是结论成"缺的是滞空时间"。**这个推理有循环论证**：滞空时间是用
"旋转爆发持续多久"近似的，而旋转爆发持续时间本身就由转速和转数共同决定。
真正独立的滞空度量需要触地时刻，而手动日志里 `real_f_pin_*` 全为 0，拿不到。

**第二版（正确）**：自动评测侧有独立的 `first_air_t` / `landing_t`，可以做反事实。
但直接算 `angle / (landing_t − first_air_t)` 也不行——`landing_t` 是**足底接触**，
仰面落地时躯干先着地、足底后着地，这个"滞空时间"会被拉长到 1.4–2.0 s（顶端
1.0 m 的飞行不可能飞 2 s）。所以先筛出弹道自洽的飞行：
`air / (2√(2(apex−0.28)/g))` 落在 0.85–1.15。前后双翻 45 次里只有 12 次通过。

在这个干净子集上：

| 技能 | 干净样本 | 说明 | 平均滞空 | 转速 |
|---|---:|---|---:|---:|
| front-2 | 12 | 全部完成 2 转 | 0.88 s | 14.4 rad/s |
| front-2 | 33 | 被筛掉（非足底先着地） | — | — |
| back-2 | 15 | 完成 2 转 | 0.86 s | 14.3 rad/s |
| back-2 | 7 | 未完成 | 0.86 s | 13.5 rad/s |
| left-2 | 43 | 全部完成 | 0.81 s | 15.4 rad/s |
| right-2 | 36 | 全部完成 | 0.80 s | 15.6 rad/s |

**反事实检验**（`back-2` 的 7 次未完成）：

- 给它们已完成组的转速 14.3 rad/s → **4 次能转完**；
- 给它们已完成组的滞空 0.86 s → **0 次能转完**（它们本来就有 0.86 s）。

结论：**转速差是主因，滞空时间不是。** 前后双翻与左右双翻的滞空几乎相同
（0.86 vs 0.80 s），转速差约 1 rad/s（7%），正好横跨判定边界。

用轨迹里的陀螺仪（`omega`，机体坐标，2 ms 采样）独立验证，不依赖任何接触检测。
取 `|ω|` 的样本均值（不先求和再取绝对值，否则正负抵消）。`hold` 列是飞行中
转速达到"该次飞行所需转速的 80%"的时间占比：

| 技能 | 组 | n | 滞空 | 需要 | \|ω_y\| 均值 | ω_y p90 | hold_y | tilt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| front-1 | 完成 | 99 | 0.87 s | 7.3 | 7.40 | 7.8 | 0.97 | 10.5° |
| front-2 | 完成 | 23 | 1.06 s | 12.6 | 12.56 | 15.4 | 0.80 | 15.1° |
| front-2 | 未完成 | 14 | 1.28 s | 10.7 | **9.98** | 14.3 | **0.68** | 83.6° |
| back-2 | 完成 | 18 | 0.88 s | 14.5 | 13.67 | 16.1 | 0.83 | 17.4° |
| back-2 | 未完成 | 25 | 1.10 s | 12.1 | **10.72** | 14.9 | **0.67** | 64.2° |
| left-2 | 完成 | 43 | 0.79 s | 15.9 | (ω_x 15.34) | 19.6 | 0.77 | 12.0° |
| right-2 | 完成 | 43 | 0.78 s | 16.0 | (ω_x 15.68) | 19.3 | 0.76 | 13.8° |

两点值得注意：

1. **完成的组均值与需求几乎相等**（front-2: 12.56 vs 12.6；back-2: 13.67 vs 14.5），
   也就是**零余量**。这不是"有余量但偶尔失手"，是"刚好够、稍掉就失败"。
2. **未完成的组 p90 仍有 14.3–14.9 rad/s**，但均值只有 10.0–10.7，`hold` 掉到
   0.67–0.68。也就是说**峰值能打到，但维持不住**——转速在中段掉下去，
   后段补不回来，最后差 30–60°。

左右翻正好相反：`hold_x` 0.76–0.77，`p90` 19.3–19.6 rad/s，均值 15.3–15.7，
比"需求"还高一点点，所以 43/43 全过。

陀螺仪积分与评测器自己记录的 `angle` 对得上（中位差 −0.02 到 −0.13 圈），
所以 `angle` 这个评分量本身可信，不是接触检测的伪影。

时间预算（门限 1.30 m，站立 base 高约 0.28 m）：

| 顶点 | 滞空 | 转完两圈需要 |
|---:|---:|---:|
| 1.00 m | 0.77 s | 16.4 rad/s |
| 1.10 m | 0.82 s | 15.4 rad/s |
| **1.30 m（门限）** | **0.91 s** | **13.8 rad/s** |
| 1.50 m | 1.00 s | 12.6 rad/s |

实测前后双翻的**飞行内平均**俯仰转速：完成的 12.6–13.7 rad/s，未完成的
**9.4–10.3 rad/s**；左右翻 15.3–15.7 rad/s。门限对应的需求是 13.8 rad/s，
正好把前后双翻分成两半。这解释了为什么自动评测里 `left-2`/`right-2` 各自
43/43 通过，而 `front-2` 只有 20/45、`back-2` 只有 14/45。

### 手动日志与自动评测是否矛盾

不矛盾，而且互相印证：

| 量 | 手动 CSV | 自动评测 |
|---|---|---|
| 前后翻峰值转速 | 11.7–15.3 rad/s | 14.5–16.4 rad/s |
| 左右翻 | 3 次纯侧翻，全部 2.0 转、落地 3.5–5.9° | 15.3–15.7 rad/s 均值，43/43 通过 |
| 失败形态 | 目标 2 转，实际 1.5 转，落地倾角 = 缺口角 | 同样 1.4–1.9 转，落地倾角 64–84° |
| 背部着地 | 19 次双翻尝试中 10 次 | `front-2` 4 次、`back-2` 2 次 |

手动侧"目标 2 转只转 1.5 转"这一簇有 17 次，**0 次站住**，与自动侧
`rotation_angle` 失败（`back-2` 19 次，均值 1.73 转）是同一个现象。
手动日志里那些"转了 1.5 圈"的事件，`缺口角` 与 `落地倾角` 几乎相等
（132°→116°、146°→152°、169°→172°、176°→174°、180°→175°），
说明**落地角度就是没转完的那一部分**，不是转过去又弹回来。

### 落地判定的一致性问题该怎么修

两边阈值一致，所以"deploy 侧看起来更好"不是判定宽松造成的错觉，而是：

1. **训练侧会终止，deploy 侧不会。** 训练里 45° 以上直接 `task_failure`，
   策略学到的是"宁可保守着陆也别仰面躺下"；deploy 里躺下只是少一个评分项，
   `episode_successes` 不受影响，策略没有这个梯度。
2. **训练侧 `rotation_success` 强制要求 `apex_height_matched`。** 一个
   顶点不够的翻转在训练里必然失败，在 deploy 里可以只丢 `height_ok`。
3. **训练侧有 `self_contact`/`flight_plane_violation`/`takeoff_timeout`，
   deploy 侧全无。** 这三项在训练里是终止，deploy 里连观测都没有。

如果要让 deploy 的判定与训练对齐（而不是继续放宽），最小改动是：
把 `fall_max_tilt` 的 45° 终止也加进评测器的失败码，并把 `height_ok` 并入
`rotation_landing_ok`。**但这只是让报表更接近训练，不改变机器人行为**——
真正的修复必须在训练侧：**前后双翻在飞行中维持不住 13.8 rad/s 的俯仰转速**，
不是高度不够，也不是按键时机。这与上一节的 `pre_rotation` 结论一致：
俯仰轴要先抵消起跳时的 +1.7 rad/s 前倾，横滚轴不用。

机器可读结果：`results/trampoline/flip_rotation_accounting.json`。
一键复现：

```
cd src/legged_rl_deploy
python3 policies/go2/trampoline/eval/analyze_flip_rotation.py
```

脚本同时读手动 CSV（`plot/data/data.csv`）与 `results/trampoline/*/evaluator.log`
和对应 trace，输出上面所有表。

### 人眼顶点时机抖动有多大影响

单翻在空中停留 0.33 s、双翻约 0.33 s（起跳后到触床），顶点附近 ±0.1 m 的时间窗只有
约 0.23–0.34 s，±0.05 m 只有 0.16–0.24 s。以 `dvz/dt = -9.81 m/s²` 换算：

- 提前 150 ms 按 ≈ 切换时 `vz = +1.5 m/s`（还在上升），顶点低约 0.115 m；
- 滞后 150 ms ≈ 切换时 `vz = -1.5 m/s`。

评测器的 `apex` 口径落在 `vz` 刚过 0 的样本，即比顶点晚 4–6 ms——人手不可能这么准。
为量化这件事，评测器新增 `--apex-vz`：把触发阈值从 0 移开，等效于把整条切换时序
整体提前/滞后。三个臂（`jitter_early_vz15` = +1.5、`jitter_late_vz-15` = −1.5、
未改的 `apex_ab_11250_n5` = 0）使用同一份 `policy.yaml`、同一 SHA256、同一
`simulator.yaml`，是单变量对照。

扫描结果见下一节。

### 顶点时机抖动扫描（`jitter_early_vz15` / `apex_ab_11250_n5` / `jitter_late_vz-15`）

三个臂各 50 episode（5 trials × 10 技能），同一份 `policy.yaml`、同一 ONNX SHA256
（`d6624a57…`）、同一 `simulator.yaml`、同一 commit，唯一变量是 `--apex-vz`：
`+1.5`（约提前 150 ms，仍在上升）、`0`（顶点后 4–6 ms）、`−1.5`（约滞后 150 ms）。

| 口径 | 提前 150 ms | 顶点 | 滞后 150 ms |
|---|---:|---:|---:|
| 单翻 `rotation_landing_ok` (±45°) | 17/20 (85%) | 18/20 (90%) | 16/20 (80%) |
| 单翻落地倾角 ≤30° | 17/20 (85%) | 18/20 (90%) | 16/20 (80%) |
| 单翻"没摔" (≤60°) | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) |
| 双翻 `rotation_landing_ok` | 12/20 (60%) | 11/20 (55%) | 11/20 (55%) |
| 双翻落地倾角 ≤30° | 12/20 (60%) | 11/20 (55%) | 11/20 (55%) |
| 双翻"没摔" (≤60°) | 15/20 (75%) | 18/20 (90%) | 16/20 (80%) |
| 双翻平均旋转误差 (rad) | −0.76 | −0.50 | −0.90 |
| 单翻平均旋转误差 (rad) | −0.16 | −0.17 | −0.18 |

**单翻对 ±150 ms 的时机抖动不敏感**：85/90/80% 三点在 20 个样本的二项噪声范围内
（约 ±9 个百分点），且"没摔"是 100% 稳定。用户感觉到的随机性不来自这里。

**双翻对时机更敏感，但方向不对称**：提前 150 ms 时"没摔"从 90% 掉到 75%，滞后时
80%；平均旋转误差从 −0.50 分别恶化到 −0.76 / −0.90。也就是说**双翻更怕晚按**（旋转
欠得更厉害），也怕早按（落地更乱）。这与双翻目标 12.57 rad、容差只有 ±0.785 rad
（±6.3%）一致，而单翻同样容差对应 ±12.5%。

**方向排序与时机无关，是姿态造成的。** 逐方向看 `rotation_landing_ok`：

| 技能 | 提前 150 ms | 顶点 | 滞后 150 ms |
|---|---|---|---|
| front-1 | 5/5 | 5/5 | 5/5 |
| back-1 | 2/5 | 3/5 | 1/5 |
| right-1 | 5/5 | 5/5 | 5/5 |
| left-1 | 5/5 | 5/5 | 5/5 |
| front-2 | 2/5 | 1/5 | 0/5 |
| back-2 | 0/5 | 0/5 | 1/5 |
| right-2 | 5/5 | 5/5 | 5/5 |
| left-2 | 5/5 | 5/5 | 5/5 |

左右翻在三个时机臂里都是 20/20，完全不参与这场抖动。失败全部集中在前后翻，且
双翻的前后比单翻更差。这排除了"用户按键时机随机"作为前后翻差的主因——同一批按键
时机下左右翻一次没失手。

**逐条回答用户的问题：**

1. **环境是否一致？** 场景、物理 2 ms、低层配置、策略 ONNX 全部一致。
   `simulator_actuator_torque_limits:=true` 已是空操作（代码/配置/资产已删）可删掉。
   不一致的是：`evaluation_mode` 关（无状态服务与 episode 边界）、实体手柄 vs 合成
   按键、观众窗口开（只改墙钟）、以及判定口径。
2. **为什么你看到前最差、后其次、且远低于自动数字？** 前/后翻差是**真的**，机制是
   hop 顶点仍带 ~1.7 rad/s 俯仰（后翻要抵掉、前翻要借它，左右翻没有这个分量）。
   "远低于自动数字"主要是口径：自动的严格 `success` 现在也是 **单翻 1/20、双翻 0/20**，
   因为**全部卡在顶点高度 1.3 m**；若按人眼"落地没摔"算，自动是单翻 100%、双翻 90%。
   用户的主观成功率更接近后一个口径，且未统计顶点高度。
3. **多少差距来自人眼顶点抖动？** 单翻 0（±150 ms 内 80–90% 无显著差异）；
   双翻约 15 个百分点的"没摔"率（90% → 75–80%）和 +0.3 rad 的平均旋转误差。

### 前空翻单圈的时机单变量扫描（`front1_*`，2026-11-11）

用户反馈"手动前翻单圈失败率就很高"。为了把"时机"这一个变量彻底钉住，只跑
`--skills front-1`，沿 `--apex-vz` 从 +3.0 扫到 −4.5，每个臂 5–10 episode。
所有臂同一份 `policy.yaml`（`d6624a57…`）、同一 `simulator.yaml`、同一 commit。

| 臂（等效切换时刻） | n | `rotation_landing_ok` | 倾角≤30° | 顶点合格 | 平均顶点 | 平均转角 |
|---|---:|---:|---:|---:|---:|---:|
| +3.0（≈提前 300 ms） | 10 | **10** | 10 | 0 | 1.07 | 6.20 |
| +1.5（≈提前 150 ms） | 5 | 5 | 5 | 0 | 1.05 | 6.19 |
| 0（顶点，n=5） | 5 | 5 | 5 | 0 | 1.12 | 6.10 |
| 0（顶点，n=10 复测） | 10 | **10** | 10 | 0 | 1.13 | 6.08 |
| −1.5（≈滞后 150 ms） | 5 | 5 | 5 | 0 | 1.15 | 6.23 |
| −3.0（≈滞后 300 ms） | 10 | 9 | 9 | 0 | 1.14 | 5.86 |

**在 ±300 ms 的按键误差内，单圈前翻的旋转+落地判定是 39/40 ≈ 98%。**
失败只出现在滞后 ≥300 ms 一侧，且是一次 `no_landing_before_timeout`（转角只到 3.18 rad，
没转完），不是"转过了"。

`−4.5` 臂是**评测器伪影**，不是机器人行为：`apex` 触发要求 `vz ≤ −4.5 m/s`，而 entry hop
在下降段的最大速度中位数只有 ≈ −3.3 m/s，所以触发条件永不成立，评测器一直重试 entry hop
（每个 episode 记录 9 个 `role=entry` 的 hop）。该臂不产生任何 target 飞行，不参与统计。

结论：**顶点时机解释不了"手动前翻单圈失败率高"**。自动侧单圈前翻在 ±300 ms 内几乎不失误，
且失误形态是"转不完"，不是"摔"。

### 手动侧可能失败在哪里：指令交接时刻，而不是翻转本身

上面的扫描已经把"翻转质量"这一项排除了。剩下的差别与**指令交接的时机**有关。

训练的 `command_chain` 在飞行末段就把下一条指令"揭示"出来（`reveal_lead_time_range`
= 0–1 s，配合 `reveal_contact_height` 的 `time_to_contact()` 判定），并在**触地那一拍**
完成交接（`_update_command_reveal`）。也就是说策略在落地瞬间看到的指令**已经换成下一条**。

自动评测保持同一性质：`--switch-phase apex`（默认）在顶点后 4–6 ms 就发出下一条指令，
`--full-chain` 发 hop、否则 0.3 s 后按 `SELECT` 回 FixStand。

手动按"落地后再决定下一步"的习惯操作时，落地瞬间策略看到的仍是
`one_flip=1, height=1.3` 的**旧指令**——这与训练分布不一致，量级足够造成落地后失稳，
而人眼会把这一段记为"这一跳失败了"。

要强调的边界：**这一条还没有被测量**。它只是目前唯一与"翻转质量无关、且能在手动侧
解释失败率"的机制。验证方式很便宜，一次手动即可：

- 同一跳落地后立刻按 `B`（显式给 stand 指令）；
- 对照：落地后完全不按，观察是否自己再次起跳/失稳。

若后者明显更差，就确认是指令粘滞（`trampoline_command_` 只在新的按键边沿或 slot
`reset()` 时改写，落地本身不改写它），而不是单圈前翻翻不过去。

**另一个需要先对齐的前提**：手动侧若没开 `evaluation_mode`，就没有 reset/start 服务和
episode 边界，起跳点由人自己决定。评测每次 `ResetEpisode` + `FixStand` + 3 s 站姿后从
原点起跳；手动连续试跳会带着上一跳的累积位移和姿态继续（评测记录的 episode 末
XY 偏移可达 0.15–1.75 m）。这不改变物理，但改变"这一跳的初始条件"，也是主观失败率
偏高的一个来源。

### ppo02b 首轮 A/B（快照 `20261111T124424Z-974d209`）

`initial_stand` 生效后，续训前 130 个迭代相对 ppo01 同区间（11270–11370）的变化：

| 指标 | ppo01 | ppo02b | Δ |
|---|---:|---:|---:|
| `stand/success_rate` | 0.8610 | 0.9096 | **+0.049** |
| `stand_hop/success_rate` | 0.9066 | 0.9283 | +0.022 |
| `episode/survival_rate` | 0.5122 | 0.5588 | +0.047 |
| `primitive/frontflip-2/success_rate` | 0.7241 | 0.7772 | +0.053 |
| `primitive/frontflip-1/success_rate` | 0.8592 | 0.8744 | +0.015 |
| `failure/landing_fall` | 0.1200 | 0.0933 | −0.027 |
| `episode/successful_flights` | 9.129 | 7.899 | **−1.230** |
| `primitive/h/success_rate` | 0.9063 | 0.8945 | −0.012 |
| `failure/stand_drift` | 0.0001 | 0.0087 | +0.009 |

`stand` 成功率提升 4.9 个百分点是 `initial_stand` 的直接目标，符合预期；`frontflip-2`
提升 5.3 个点、`landing_fall` 下降 2.7 个点也同向。

**但这不是单变量实验**：两个快照之间除 `initial_stand`（替换 `hold_in_wait`）外，还新增了
奖励项 `hop_flight_calf_vel_l2: -1.0`（飞行中 calf 关节速度惩罚，约为此前同一速度下
sum 项的 9 倍）。因此上面的改善不能全部归给 `initial_stand`，需要在同一快照上关掉该
奖励项再跑一轮才能拆开。`successful_flights` 下降 1.23 很可能与该奖励项抑制了飞行腿
动作有关，需要在完整链上确认是否影响翻转质量。

判定门槛保持不变：仍以 sim2sim 的 hop 顶点是否达到 1.3 m 与完整链是否通过为准，
训练侧 survival 只作趋势参考。

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
