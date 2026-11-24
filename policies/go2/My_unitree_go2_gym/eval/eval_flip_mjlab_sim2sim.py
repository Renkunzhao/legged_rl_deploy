#!/usr/bin/env python3
"""Automated 2 ms Sim2Sim evaluator for all 7 origin/flip_mjlab Go2 skills.

Runs each skill in an isolated loopback ROS 2 domain (`ROS_DOMAIN_ID` != 0)
with flat rigid ground (`scene.xml`, 2 ms physics) and captures full 500 Hz
(`/lowstate`, `/lowcmd`, `/sportmodestate`) telemetry from policy activation for
torque-speed (T-omega) envelope verification.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from legged_rl_deploy.msg import DeployStatus
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowCmd, LowState, SportModeState
from unitree_mujoco.msg import EpisodeStatus

DEPLOY_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = DEPLOY_ROOT.parents[1]
sys.path.insert(0, str(DEPLOY_ROOT / "eval"))

from core.driver import ControllerDriver  # noqa: E402
from core.env import validate_simulation_environment  # noqa: E402
from core.errors import EvaluationError, SafetyStop, TransitionTimeout, TrialFailure  # noqa: E402
from core.keys import KEY_SELECT  # noqa: E402
from core.metrics import body_tilt  # noqa: E402
from core.node import EvaluationNode  # noqa: E402
from core.util import positive_float, stamp_to_ns  # noqa: E402

SKILLS = (
    "back_flip",
    "front_flip",
    "left_flip",
    "right_flip",
    "jump_up",
    "jump_forward",
    "jump_back",
)
MOTOR_COUNT = 12
SIM_CONFIG = WORKSPACE_ROOT / "src/unitree_mujoco/simulate/config/go2.yaml"
SIM_EVAL = WORKSPACE_ROOT / "src/unitree_mujoco/simulate/config/eval_trampoline.yaml"


class FlipMjlabNode(EvaluationNode):
    """500 Hz telemetry recorder built on the shared EvaluationNode."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(50.0, args.status_timeout, "flip_mjlab_sim2sim_eval")
        self.low: LowState | None = None
        self.lowcmd: LowCmd | None = None
        self.sport: SportModeState | None = None
        self.state_received_at: float | None = None
        self.recording = False
        self.last_tick: int | None = None
        self.raw_samples: list[dict[str, Any]] = []

        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.low_sub = self.create_subscription(
            LowState, "/lowstate", self._on_low, reliable_qos
        )
        self.cmd_sub = self.create_subscription(
            LowCmd, "/lowcmd", self._on_cmd, reliable_qos
        )
        self.sport_sub = self.create_subscription(
            SportModeState, "/sportmodestate", self._on_sport, reliable_qos
        )

    def _on_cmd(self, msg: LowCmd) -> None:
        self.lowcmd = msg

    def _on_sport(self, msg: SportModeState) -> None:
        self.sport = msg
        self.state_received_at = time.monotonic()

    def _on_low(self, msg: LowState) -> None:
        self.low = msg
        if not self.recording:
            return
        tick = int(msg.tick)
        if self.last_tick is not None and tick == self.last_tick:
            return
        self.last_tick = tick

        q = [float(m.q) for m in msg.motor_state[:MOTOR_COUNT]]
        dq = [float(m.dq) for m in msg.motor_state[:MOTOR_COUNT]]
        tau_est = [float(m.tau_est) for m in msg.motor_state[:MOTOR_COUNT]]

        sample: dict[str, Any] = {
            "tick": tick,
            "t_sim": tick * 1.0e-3,
            "q": q,
            "dq": dq,
            "tau_est": tau_est,
            "foot_force": [int(x) for x in msg.foot_force[:4]],
            "quat": [float(x) for x in msg.imu_state.quaternion],
            "gyro": [float(x) for x in msg.imu_state.gyroscope],
            "ctrl_state": (
                int(self.deploy_status.controller_state)
                if self.deploy_status is not None
                else -1
            ),
            "safety_ok": (
                bool(self.deploy_status.safety_ok)
                if self.deploy_status is not None
                else True
            ),
        }

        if self.lowcmd is not None:
            cmds = self.lowcmd.motor_cmd[:MOTOR_COUNT]
            sample["kp"] = [float(m.kp) for m in cmds]
            sample["kd"] = [float(m.kd) for m in cmds]
            sample["q_des"] = [float(m.q) for m in cmds]
            sample["tau_ff"] = [float(m.tau) for m in cmds]
            sample["tau_pd_raw"] = [
                float(m.kp) * (float(m.q) - q[i]) + float(m.kd) * (float(m.dq) - dq[i])
                for i, m in enumerate(cmds)
            ]
            sample["tau_cmd_applied"] = [
                float(m.tau) + sample["tau_pd_raw"][i] for i, m in enumerate(cmds)
            ]

        if self.sport is not None:
            feet = np.asarray(self.sport.foot_position_body, dtype=float).reshape(4, 3)
            pos = [float(x) for x in self.sport.position]
            vel = [float(x) for x in self.sport.velocity]
            sample["position"] = pos
            sample["velocity"] = vel
            sample["body_height"] = float(pos[2] - np.mean(feet[:, 2]))

        self.raw_samples.append(sample)

    def extra_freshness(self) -> dict[str, float | None]:
        return {"robot state": self.state_received_at}

    def preflight_extra_ready(self) -> bool:
        return self.low is not None and self.sport is not None and self.lowcmd is not None


def trim_high_controller_samples(
    raw_samples: list[dict[str, Any]], duration_s: float
) -> tuple[list[dict[str, Any]], str]:
    """Slice exact 500 Hz samples starting at policy takeover."""
    start_idx = None
    detect_mode = "kp_drop"
    for idx, s in enumerate(raw_samples):
        kp = s.get("kp")
        if kp is not None and len(kp) >= 3 and kp[0] < 30.0 and kp[2] < 72.0:
            start_idx = idx
            break
    if start_idx is None:
        detect_mode = "deploy_status"
        for idx, s in enumerate(raw_samples):
            if s.get("ctrl_state") == DeployStatus.CONTROLLER_STATE_HIGH_CONTROLLER:
                start_idx = idx
                break
    if start_idx is None:
        raise RuntimeError("never detected policy takeover in recorded samples")

    t0 = raw_samples[start_idx]["t_sim"]
    trimmed: list[dict[str, Any]] = []
    for s in raw_samples[start_idx:]:
        t_rel = s["t_sim"] - t0
        if t_rel > duration_s + 1.0e-6:
            break
        row = dict(s)
        row["t_rel"] = round(t_rel, 6)
        trimmed.append(row)
    return trimmed, detect_mode


def run_single_skill(skill: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run handshake + 4 s rollout for one skill inside an active ROS graph."""
    rclpy.init(args=[])
    node = FlipMjlabNode(args)
    driver = ControllerDriver(node, args)
    trace_path = args.output / "traces" / f"{skill}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        node.wait_for_preflight(args.preflight_timeout)
        node.publishing_enabled = True
        driver.normalize_to_fixed_stand()

        prev_ep = node.episode_status.episode_id
        resp = node.call_reset(
            args.service_timeout,
            base_x_offset=0.0,
            base_y_offset=0.0,
            base_yaw_offset=0.0,
        )
        if resp.episode_id <= prev_ep:
            raise EvaluationError(
                f"ResetEpisode did not advance episode_id ({prev_ep} -> {resp.episode_id})"
            )
        reset_stamp_ns = stamp_to_ns(resp.reset_stamp)
        reset_wall = time.monotonic()

        def reset_ready() -> bool:
            ep = node.episode_status
            dep = node.deploy_status
            return (
                ep.episode_id == resp.episode_id
                and ep.outcome == EpisodeStatus.WAITING
                and stamp_to_ns(ep.stamp) > reset_stamp_ns
                and abs(ep.roll) <= 0.15
                and abs(ep.pitch) <= 0.15
                and ep.angular_velocity_rad_s <= 0.30
                and ep.linear_velocity_m_s <= 0.05
                and dep.safety_ok
                and dep.controller_state == DeployStatus.CONTROLLER_STATE_FIX_STAND
                and dep.fix_stand_ready
                and (time.monotonic() - reset_wall) >= args.settle_time
            )

        try:
            node.wait_until(
                reset_ready,
                timeout=args.reset_ready_timeout,
                description="reset episode settled in FIX_STAND",
                fail_on_estop=True,
            )
        except SafetyStop as exc:
            raise TrialFailure("DEPLOY_ESTOP", str(exc)) from exc
        except TransitionTimeout as exc:
            raise TrialFailure("RESET_READY_TIMEOUT", str(exc)) from exc

        # Start recording immediately BEFORE sending KEY_START so step 0 is captured.
        node.raw_samples.clear()
        node.last_tick = None
        node.recording = True
        driver.enter_policy()
        driver.arm_episode(resp.episode_id)

        wall_deadline = time.monotonic() + args.duration * 4.0 + 10.0
        target_t: float | None = None
        while rclpy.ok() and time.monotonic() < wall_deadline:
            node.spin_once(0.0005)
            node.assert_fresh()
            if not node.deploy_status.safety_ok:
                break
            if target_t is None:
                for s in node.raw_samples:
                    kp = s.get("kp")
                    if kp is not None and len(kp) >= 3 and kp[0] < 30.0 and kp[2] < 72.0:
                        target_t = s["t_sim"] + args.duration
                        break
            if target_t is not None and node.raw_samples:
                if node.raw_samples[-1]["t_sim"] >= target_t:
                    break

        node.recording = False
        trimmed, detect_mode = trim_high_controller_samples(node.raw_samples, args.duration)
        with trace_path.open("w", encoding="utf-8") as fp:
            for row in trimmed:
                fp.write(json.dumps(row) + "\n")

        driver.pulse(KEY_SELECT)
        node.set_command()

        tilts = [body_tilt(s["quat"]) for s in trimmed]
        zs = [s["position"][2] for s in trimmed if "position" in s]
        dq_max = max(max(abs(v) for v in s["dq"]) for s in trimmed)
        tau_est_max = max(max(abs(v) for v in s["tau_est"]) for s in trimmed)
        tau_pd_max = max(
            max(abs(v) for v in s["tau_pd_raw"])
            for s in trimmed
            if "tau_pd_raw" in s
        )
        return {
            "skill": skill,
            "trace_path": str(trace_path),
            "samples": len(trimmed),
            "detect_mode": detect_mode,
            "duration_sim_s": trimmed[-1]["t_rel"] if trimmed else 0.0,
            "safety_ok": bool(node.deploy_status.safety_ok),
            "z_min_m": min(zs) if zs else None,
            "z_max_m": max(zs) if zs else None,
            "z_end_m": zs[-1] if zs else None,
            "max_tilt_deg": float(np.degrees(max(tilts))) if tilts else None,
            "end_tilt_deg": float(np.degrees(tilts[-1])) if tilts else None,
            "max_abs_dq": float(dq_max),
            "max_abs_tau_est": float(tau_est_max),
            "max_abs_tau_pd_raw": float(tau_pd_max),
        }
    finally:
        node.recording = False
        if node.publishing_enabled and rclpy.ok():
            try:
                node.set_command()
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def kill_proc_group(proc: subprocess.Popen[bytes] | None) -> None:
    """Terminate an isolated process group cleanly, falling back to SIGKILL."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def build_sim_env(domain_id: int) -> dict[str, str]:
    """Return environment dict preserving isolated loopback ROS 2 settings."""
    env = os.environ.copy()
    env["WORKSPACE"] = str(WORKSPACE_ROOT)
    env["NetworkInterface"] = "lo"
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    env["CYCLONEDDS_URI"] = (
        '<CycloneDDS><Domain><General><Interfaces>'
        '<NetworkInterface name="lo" multicast="false" />'
        '</Interfaces></General></Domain></CycloneDDS>'
    )
    torch_lib = str(DEPLOY_ROOT / "thirdparty/libtorch/lib")
    prev_ld = env.get("LD_LIBRARY_PATH", "")
    if torch_lib not in prev_ld.split(":"):
        env["LD_LIBRARY_PATH"] = f"{torch_lib}:{prev_ld}" if prev_ld else torch_lib
    return env


def orchestrate(args: argparse.Namespace) -> None:
    """Launch sim + controller per skill and aggregate summary JSON."""
    domain = validate_simulation_environment()
    env = build_sim_env(domain)
    args.output.mkdir(parents=True, exist_ok=True)
    log_dir = args.output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for skill in args.skills:
        policy_rel = f"src/legged_rl_deploy/policies/go2/My_unitree_go2_gym/{skill}/config.yaml"
        if not (WORKSPACE_ROOT / policy_rel).is_file():
            raise FileNotFoundError(f"Missing policy config: {WORKSPACE_ROOT / policy_rel}")

        sim_log = (log_dir / f"{skill}.sim.log").open("w", encoding="utf-8")
        ctl_log = (log_dir / f"{skill}.ctrl.log").open("w", encoding="utf-8")
        sim_proc: subprocess.Popen[bytes] | None = None
        ctl_proc: subprocess.Popen[bytes] | None = None
        try:
            sim_cmd = [
                "ros2", "run", "unitree_mujoco", "unitree_mujoco",
                "--config", str(SIM_CONFIG),
                "--eval", str(SIM_EVAL),
                "-s", "scene.xml",
                "--no-viewer", "--no-joystick",
            ]
            if args.motor_curve == "true":
                sim_cmd.append("--motor-curve=true")
            elif args.motor_curve == "false":
                sim_cmd.append("--motor-curve=false")
            if args.motor_curve_mode is not None:
                sim_cmd.extend(["--motor-curve-mode", args.motor_curve_mode])
            if args.motor_bus_voltage is not None:
                sim_cmd.extend(["--motor-bus-voltage", str(args.motor_bus_voltage)])
            ctl_cmd = [
                "ros2", "run", "legged_rl_deploy", "legged_rl_deploy_node",
                "lo", policy_rel,
                "--ros-args", "-p", "evaluation_mode:=true",
            ]
            sim_proc = subprocess.Popen(
                sim_cmd, cwd=WORKSPACE_ROOT, env=env,
                stdout=sim_log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
            ctl_proc = subprocess.Popen(
                ctl_cmd, cwd=WORKSPACE_ROOT, env=env,
                stdout=ctl_log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )

            worker_cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--single-skill", skill,
                "--output", str(args.output),
                "--duration", str(args.duration),
                "--settle-time", str(args.settle_time),
                "--reset-ready-timeout", str(args.reset_ready_timeout),
                "--preflight-timeout", str(args.preflight_timeout),
                "--button-hold", str(args.button_hold),
                "--status-timeout", str(args.status_timeout),
                "--service-timeout", str(args.service_timeout),
                "--transition-timeout", str(args.transition_timeout),
            ]
            res = subprocess.run(
                worker_cmd, cwd=WORKSPACE_ROOT, env=env,
                capture_output=True, text=True, timeout=90,
            )
            if res.stdout:
                print(res.stdout.strip(), flush=True)
            if res.returncode != 0:
                raise RuntimeError(
                    f"Skill {skill} worker failed (rc={res.returncode}):\n{res.stderr}"
                )
            skill_summary = json.loads(
                (args.output / f"{skill}.summary.json").read_text(encoding="utf-8")
            )
            summaries.append(skill_summary)
        finally:
            kill_proc_group(ctl_proc)
            kill_proc_group(sim_proc)
            sim_log.close()
            ctl_log.close()

    motor_curve_args: dict[str, Any] = {
        "motor_curve": args.motor_curve,
        "motor_curve_mode": args.motor_curve_mode,
        "motor_bus_voltage": args.motor_bus_voltage,
    }
    (args.output / "sim2sim_run_summary.json").write_text(
        json.dumps({"ros_domain": domain, "motor_curve_args": motor_curve_args, "skills": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skills", nargs="+", choices=SKILLS, default=list(SKILLS))
    parser.add_argument("--single-skill", choices=SKILLS, default=None)
    parser.add_argument("--duration", type=positive_float, default=4.0)
    parser.add_argument("--settle-time", type=positive_float, default=2.0)
    parser.add_argument("--reset-ready-timeout", type=positive_float, default=10.0)
    parser.add_argument("--preflight-timeout", type=positive_float, default=20.0)
    parser.add_argument("--button-hold", type=positive_float, default=0.06)
    parser.add_argument("--status-timeout", type=positive_float, default=2.0)
    parser.add_argument("--service-timeout", type=positive_float, default=5.0)
    parser.add_argument("--transition-timeout", type=positive_float, default=8.0)
    parser.add_argument("--motor-curve", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--motor-curve-mode", choices=("dc_motor", "motoring_only"), default=None)
    parser.add_argument("--motor-bus-voltage", type=positive_float, default=None)
    args = parser.parse_args()

    if args.single_skill is not None:
        validate_simulation_environment()
        summary = run_single_skill(args.single_skill, args)
        (args.output / f"{args.single_skill}.summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary), flush=True)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
