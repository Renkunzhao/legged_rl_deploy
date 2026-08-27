# Automated Beam Sim-to-Sim Evaluation

## Build And Environment

In every terminal, select loopback first and then override the domain. The
`unitree_lowlevel` setup script sets domain 0, so the override must come after
it:

```bash
cd /home/rkz/code/unitree_ws
source src/unitree_lowlevel/scripts/setup.sh lo jazzy
export ROS_DOMAIN_ID=1
```

```bash
ros2 run unitree_mujoco unitree_mujoco \
  -r go2 -s scene_bridge.xml \
  --depth-camera --no-viewer --no-joystick --beam-monitor \
  --episode-timeout 20.0
```

`--no-viewer` skips the main render loop but still uses GLFW for the hidden
OpenGL context and offscreen depth camera. A working `DISPLAY` is therefore
still required; this is not an EGL display-server-free mode.

Start the depth preprocessor in a second terminal with the same environment:

```bash
ros2 run legged_rl_deploy depth_image_preprocessor_node.py \
  --ros-args --params-file \
  src/legged_rl_deploy/policies/go2/unitree_rl_mjlab/beam_depth_distillation/depth_image_preprocessor.yaml
```

After warm-up, measure each stream separately with a 300-frame window:

```bash
ros2 topic hz /camera/depth/image_rect_raw --window 300
ros2 topic hz /unitree_go2_beam_depth/depth_m --window 300
```

```bash
ros2 run legged_rl_deploy legged_rl_deploy_node lo \
  src/legged_rl_deploy/policies/go2/unitree_rl_mjlab/beam_depth_distillation/config.yaml \
  --ros-args -p evaluation_mode:=true
```

## Run Trials

```bash
python3 src/legged_rl_deploy/scripts/eval_beam_sim2sim.py \
  --trials 128 \
  --velocity 0.6 \
  --seed 42 \
  --reset-position-jitter-m 0.10 \
  --reset-yaw-jitter-rad 0.20
```

The evaluator derives both output names from the experiment parameters and
writes them in the current directory:

```text
eval-sim2sim-vx0.6-reset-pos0.1m-reset-yaw0.2rad-seed42-n128.jsonl
eval-sim2sim-vx0.6-reset-pos0.1m-reset-yaw0.2rad-seed42-n128.summary.json
```

Existing files are never overwritten.

## Episode Sequence

For each trial the evaluator performs this handshake:

1. Publish neutral and establish a ready `FIX_STAND`. From `IDLE`, this uses an
   `L2+A` press and waits for `fix_stand_ready`; from `HIGH_CONTROLLER`, it uses
   `SELECT` first.
2. Call `ResetEpisode` only while deploy remains in ready `FIX_STAND`.
3. Wait for the new episode id, `WAITING`, a minimum settle interval, bounded
   roll/pitch, angular and linear velocity, and reset-new depth stamps from
   MuJoCo and the processed-depth topic.
4. Press `START` and wait for `HIGH_CONTROLLER`, a strictly increased policy
   reset sequence, and valid policy output. A valid output proves deploy read a
   valid, non-stale external input, but does not claim exact image/inference
   synchronization.
5. Call `StartEpisode`, wait for `RUNNING`, and only then publish nonzero `ly`.
6. On a terminal outcome, publish neutral, press `SELECT`, and wait for ready
   `FIX_STAND` before the next reset.

If deploy reports e-stop, the evaluator records the current trial as failed,
publishes neutral, sends `SELECT+START` to clear e-stop, waits for safe `IDLE`,
and then uses `L2+A` to return to ready `FIX_STAND`. Communication loss,
competing wireless publishers, reset failures, and inconsistent episode ids
are fatal errors rather than silently skipped trials.
