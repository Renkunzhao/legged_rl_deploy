# Automated sim2sim evaluation

Shared evaluation layer for deployment-side tasks. It drives the real
`unitree_mujoco` + `unitree_lowlevel` + `legged_rl_deploy` stack through ROS 2; it
does not simulate physics or run policy inference itself.

```text
eval/
├── README.md
└── core/                 task-agnostic
    ├── node.py           wireless publisher, deploy/episode status, reset/start
    │                     services, heartbeat, staleness, single-publisher guard
    ├── driver.py         IDLE -> FIX_STAND -> HIGH_CONTROLLER, episode arming
    ├── keys.py           button bit masks
    ├── metrics.py        quaternion / tilt / yaw helpers
    ├── env.py            loopback ROS environment validation
    ├── util.py           timestamps and argparse validators
    └── errors.py         EvaluationError / TransitionTimeout / SafetyStop / TrialFailure
```

`eval/` is a library only: it holds no task. Task evaluators live with the policy
bundle they drive, so a policy directory stays self-contained:

```text
policies/go2/trampoline/eval/                                  trampoline skill chain (Python-scored)
policies/go2/unitree_rl_mjlab/beam_depth_distillation/eval/    bridge task (C++ episode monitor)
```

## Contract

- Import the shared layer by putting this directory on `sys.path` and using
  `from core.<module> import ...`.
- A task adds its own node subclass and reports task telemetry through
  `extra_freshness()`; the base class applies the same staleness policy and the
  same preflight checks.
- A task never re-implements the controller state machine; it subclasses
  `ControllerDriver`.

## Environment

Every evaluator requires an isolated loopback graph:

```bash
export ROS_DOMAIN_ID=<free non-zero id>
export NetworkInterface=lo
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" multicast="false"/></Interfaces></General></Domain></CycloneDDS>'
```

The simulator must run with `--no-joystick`: the evaluator asserts that it is the
only publisher of `/wirelesscontroller`.
