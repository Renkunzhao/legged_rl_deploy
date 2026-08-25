# Legged RL Deploy

`legged_rl_deploy` deploys learned policies on top of `unitree_lowlevel` for Unitree Go2 and G1 robots.

It provides:

- single-policy deployment from a single `config.yaml`
- multi-policy deployment with gamepad-triggered switching via `fsm`
- ONNX Runtime and LibTorch inference backends
- local-file and Redis-based mimic motion sources
- sample policies and deployment configs for Go2 and G1

## Overview

This package inherits the low-level control loop, safety logic, and hardware adapters from [`unitree_lowlevel`](../unitree_lowlevel/README.md). The RL policy only becomes active after the low-level controller enters `HighController`.

The deployed config is passed directly to:

```bash
ros2 run legged_rl_deploy legged_rl_deploy_node <network-interface> <config-file>
```

Two config styles are supported:

- Single-policy config: top-level `policy:` section
- Multi-policy config: top-level `fsm:` section with multiple sub-policies and gamepad transitions

Video demo:

- https://renkunzhao.github.io/present/slides/phd-interview/#/5

## Policy Layout

The repository already includes several example policies:

```text
policies/
├── go2/
│   ├── My_unitree_go2_gym/
│   │   ├── trot/
│   │   ├── hop/
│   │   ├── handstand/
│   │   ├── legstand/
│   │   ├── spring_jump/
│   │   └── multi-policies.yaml
│   ├── mjlab/
│   │   ├── velocity/
│   │   ├── hop/
│   │   ├── mimic/
│   │   └── multi-policies.yaml
│   ├── unitree_rl_lab/
│   │   └── velocity/
│   └── aba/
│       └── velocity/
└── g1/
    ├── velocity/
    │   ├── unitree_rl_mjlab/
    │   └── mjlab/
    ├── mimic/
    │   ├── unitree_rl_lab/
    │   │   ├── dance_102/
    │   │   └── gangnam_style/
    │   ├── unitree_rl_mjlab/
    │   │   └── dance1_subject2/
    │   ├── mjlab/
    │   │   └── dance1_subject1/
    │   ├── whole_body_tracking/
    │   │   └── jumps1_subject1/
    │   └── TWIST2/
    ├── aba/
    │   ├── velocity/
    │   ├── dance102/
    │   ├── jumps1_subject1/
    │   └── multi-policies.yaml
    └── multi-policies.yaml
```

Included policy sources referenced by this repo:

| Source | Link |
| --- | --- |
| unitree_rl_lab | [github.com/unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) |
| My_unitree_go2_gym | [github.com/yusongmin1/My_unitree_go2_gym](https://github.com/yusongmin1/My_unitree_go2_gym) |
| mjlab | [github.com/mujocolab/mjlab](https://github.com/mujocolab/mjlab) |
| unitree_rl_mjlab | [github.com/mujocolab/unitree_rl_mjlab](https://github.com/mujocolab/unitree_rl_mjlab) |

Some configs in this repo are experiment-specific derivatives or internally retargeted motions.

## Dependencies

### Required

- [`unitree_lowlevel`](https://github.com/Renkunzhao/unitree_lowlevel) built in the same workspace
- ROS 2 workspace already initialized and buildable with `colcon`

### Recommended for the bundled policies

Most bundled policies use `backend: ort`, so ONNX Runtime is effectively required for the default examples:

```bash
cd $WORKSPACE/src/legged_rl_deploy
./scripts/get_ort.sh
```

This downloads ONNX Runtime into `thirdparty/onnxruntime`.

### Optional: LibTorch

Use LibTorch only if your policy config sets `backend: libtorch` or `backend: torch`.

Helper script:

```bash
cd $WORKSPACE/src/legged_rl_deploy
./scripts/get_torch.sh
```

Platform-specific notes are in [LibTorch.md](LibTorch.md).

### Optional: Redis mimic source

Redis support is compiled in only when `hiredis` is available at build time:

```bash
sudo apt install -y libhiredis-dev redis-tools
```

This is only needed for configs using `mimic.params.source: redis`, such as `policies/g1/mimic/TWIST2/config.yaml`.

## Build

Clone the repository into the workspace:

```bash
cd unitree_ws/src
git clone https://github.com/Renkunzhao/legged_rl_deploy.git
```

If you plan to use:

- ONNX Runtime: run `./scripts/get_ort.sh` first
- LibTorch: install it and set environment variables as described in [LibTorch.md](LibTorch.md) before building
- Redis mimic: install `libhiredis-dev` before building

Then build:

```bash
cd unitree_ws
source /opt/ros/<ros-distro>/setup.bash
colcon build --packages-up-to legged_rl_deploy --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

> Note: The explicit `Python3_EXECUTABLE` avoids clean-build failures when `python3` points to a Conda environment without the ROS build dependencies.

## Running

### Hardware preparation

`legged_rl_deploy` still relies on the low-level controller setup from `unitree_lowlevel`.

For Go2 hardware, stop the default controller once after each boot:

```bash
source src/unitree_lowlevel/scripts/setup.sh <network-interface> <ros-distro>
$WORKSPACE/build/unitree_sdk2/bin/go2_stand_example <network-interface>
```

For G1 hardware, enter `Debug Mode` first as described in the [`unitree_lowlevel` documentation](../unitree_lowlevel/README.md).

### Recommended launcher

From the workspace root, use the helper wrapper:

```bash
source src/unitree_lowlevel/scripts/setup.sh <network-interface> <ros-distro>
source src/unitree_lowlevel/scripts/setup.sh lo jazzy

ros2 run legged_rl_deploy depth_image_preprocessor_node.py \
  --ros-args --params-file \
  src/legged_rl_deploy/<policy-directory>/depth_image_preprocessor.yaml

./src/legged_rl_deploy/scripts/run.sh <network-interface> <ros-distro> ros2 run legged_rl_deploy legged_rl_deploy_node <network-interface> <config-file>
```

`depth_image_preprocessor_node.py` is the only depth preprocessing executable.
Each policy YAML selects an ordered subset of `stereo_occlusion`,
`replace_invalid`, `clip`, `center_crop`, `resize_nearest`, and `affine`.
The node rejects unknown, repeated, out-of-order, missing, and unused operation
parameters at startup. The Bridge YAML includes simulation-only stereo
occlusion in the same pipeline; remove that operation and its parameter block
when using a physical stereo camera.

The wrapper:

- sources `unitree_lowlevel/scripts/setup.sh`
- sets `CMAKE_PREFIX_PATH` and `LD_LIBRARY_PATH` for Torch on x86_64
- sets the corresponding Torch paths from Python on aarch64

### Runtime behavior

The node inherits the same safety confirmation as `unitree_lowlevel`:

```text
WARNING: Make sure the robot is hung up or lying on the ground.
Press Enter to continue...
```

In practice, the flow is:

1. Start the node
2. Confirm the safety prompt
3. Use the low-level controller gamepad flow to reach `FixStand`
4. Press `START` to enter `HighController`
5. The RL policy then takes control

If you are using a multi-policy config, policy switching is handled by the `fsm.policies.*.transitions` entries in the YAML file.

## Config Notes

Single-policy configs typically define:

- `llc_config_file`: low-level config from `unitree_lowlevel`
- `ll_dt`: low-level timestep override
- `policy.backend`: `ort`, `onnxruntime`, `torch`, or `libtorch`
- `policy.model_path`: model file path relative to `$WORKSPACE`
- `policy.observations`: observation terms and assembly layout
- `policy.actions`: output post-processing
- `policy.commands`: command preprocessing

The compact `input_dim`/`output_dim` form is for a model with one observation
input and one action output. Models with any other signature declare the full
contract under `policy.model`:

- `model.inputs`: tensor `name`, concrete `shape`, and `source`
- `model.outputs`: tensor `name`, concrete `shape`, and `target`
- input sources: `observations`, `external.<name>`, `state.<name>`, or
  `constant`
- output targets: `actions`, `state.<name>`, or `discard`
- `model.states`: zero-initialized explicit loop buffers with an optional
  `max_norm`

Exactly one output must target `actions`. Every declared state must bind to one
input and one output. The runner owns these buffers and resets them whenever a
policy slot resets. A stateful TorchScript module does not declare
`model.states`; its internal state stays inside the module, and an exported
zero-argument `reset()` method is called when present.

All model inputs, outputs, constants, and explicit state buffers currently use
`float32`. ONNX or TorchScript models with integer tensor inputs are not
supported by the current runners.

ROS image inputs are configured under `policy.external_inputs`. They are strict
tensor transports and currently require a tightly packed, little-endian
`32FC1` image matching the configured `[1, 1, H, W]` model shape. The encoding
describes the final policy tensor, not the raw camera stream. Raw image decoding,
resize, clipping, invalid-value handling, and normalization belong in a
separate preprocessing node; no automatic tensor layout conversion is applied.

Multi-policy configs additionally define:

- `clip_final_tau`: optional torque-mode output instead of PD targets
- `fsm.default`: policy activated when entering `HighController`
- `fsm.policies.<name>.config`: sub-policy config path
- `fsm.policies.<name>.transitions`: button mappings for switching

## Teleop and Redis Mimic

For Redis-based mimic motion:

```bash
sudo apt install -y libhiredis-dev redis-tools
```

Rebuild `legged_rl_deploy` after installing `hiredis`, otherwise Redis support will remain disabled in the compiled binary.

You can inspect Redis keys with:

```bash
redis-cli -h 127.0.0.1 -p 6379 -n 0 --scan
```

## Troubleshooting

### `backend=onnxruntime requested, but built without USE_ORT`

Run:

```bash
cd $WORKSPACE/src/legged_rl_deploy
./scripts/get_ort.sh
```

Then rebuild the package.

### `backend=libtorch requested, but built without USE_TORCH`

Install LibTorch, export the required paths from [LibTorch.md](LibTorch.md), and rebuild the package.

### Redis mimic complains that hiredis is missing

Install `libhiredis-dev`, then rebuild the package so `USE_HIREDIS` is enabled.

### CMake CUDA dialect error

If you see an error like:

```text
Target "cmTC_xxxx" requires the language dialect "CUDA17", but CMake does not know the compile flags to use to enable it.
```

Update CMake to a newer version that supports CUDA17.
