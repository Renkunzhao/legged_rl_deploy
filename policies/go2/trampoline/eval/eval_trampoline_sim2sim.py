#!/usr/bin/env python3
"""Exercise the real ROS/C++ policy using wireless button edges and reset services.

Reuses the shared eval core's controller handshake. No policy inference is performed
here. Records raw state/command traces and scores complete takeoff/landing flights.
``--switch-phase`` selects when the next command is sent: ``apex`` (default,
manual-like), ``takeoff`` (training's reveal at liftoff, 20 ms after the first
airborne sample), or ``landing``.

The flight metric deliberately keeps the training phase convention: a grounded
heading/axis reference is frozen before the lift rotation, the anchored pre-rotation
is added once, and the flight angle is integrated with projected body-up increments
about that fixed axis. The sign is never folded with abs().
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from legged_rl_deploy.msg import DeployStatus
from unitree_go.msg import LowCmd, LowState, SportModeState
from unitree_mujoco.msg import EpisodeStatus

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "eval"))

from core.driver import ControllerDriver
from core.env import validate_simulation_environment
from core.errors import EvaluationError, SafetyStop, TransitionTimeout, TrialFailure
from core.keys import KEY_A, KEY_SELECT
from core.metrics import body_tilt, rotation, wrap_to_pi, yaw_twist
from core.node import EvaluationNode
from core.util import positive_float, positive_int, stamp_to_ns

KEY_B = 1 << 9
KEY_L1 = 1 << 1
SKILLS = {
    "hop": (KEY_A, (0., 0., 0.), 0),
    "front-1": (1 << 12, (0., 1., 0.), 1),
    "back-1": (1 << 14, (0., -1., 0.), 1),
    "right-1": (1 << 13, (1., 0., 0.), 1),
    "left-1": (1 << 15, (-1., 0., 0.), 1),
    "front-2": ((1 << 12) | KEY_L1, (0., 1., 0.), 2),
    "back-2": ((1 << 14) | KEY_L1, (0., -1., 0.), 2),
    "right-2": ((1 << 13) | KEY_L1, (1., 0., 0.), 2),
    "left-2": ((1 << 15) | KEY_L1, (-1., 0., 0.), 2),
}

# Pose tolerance for the evaluated stand, in hardware order (FR, FL, RR, RL).
STAND_NOMINAL = np.tile([.1, .9, -1.8], (4, 1))
STAND_NOMINAL[[1, 3], 0] = -.1
STAND_POSE_TOL = np.array([.2, .3, .3])
STAND_MIRROR_TOL = .2
STAND_TAIL_S = 2.0
# The ROS bridge subtracts the foot radius when it fills foot_position_body;
# base-height diagnostics are therefore above the foot BOTTOM.  Training's
# _stand_height uses foot CENTRES, so score with body_height - FOOT_RADIUS.
FOOT_RADIUS = .022
PHASE_EPS = 1.0e-3
# --switch-phase takeoff: first policy step after the flight is established.
TAKEOFF_SWITCH_DELAY_S = .02
MOTOR_COUNT = 12


def pre_rotation_about_axis(up, axis):
    """Training's pre-rotation: angle from upright to body-up in the flip plane.

    Both vectors are projected onto the plane normal to the (horizontal) flip
    axis, exactly as the training ``_normalized_projection`` does.  Returns
    ``(angle, valid)``; invalid when the body-up is (nearly) along the flip
    axis, where the training reference is also invalid.
    """
    axis = np.asarray(axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1.0e-9:
        return 0.0, True
    axis = axis / axis_norm
    up = np.asarray(up, dtype=float)
    projected_up = up - axis * float(np.dot(up, axis))
    up_norm = float(np.linalg.norm(projected_up))
    upright = np.array([0.0, 0.0, 1.0])
    projected_upright = upright - axis * float(np.dot(upright, axis))
    upright_norm = float(np.linalg.norm(projected_upright))
    if up_norm < PHASE_EPS or upright_norm < PHASE_EPS:
        return 0.0, False
    ref = projected_up / up_norm
    base = projected_upright / upright_norm
    return math.atan2(float(np.dot(axis, np.cross(base, ref))),
                       float(np.dot(base, ref))), True


def phase_increment(up_prev, up_curr, axis):
    """Projected body-up phase increment about a fixed flight axis.

    This is the training phase metric: project both body-up vectors into the
    plane perpendicular to the axis and take the signed angle between them.
    Returns ``(increment, valid)``; the caller must keep the last valid
    reference when invalid so a degenerate sample does not silently reset it.
    """
    axis = np.asarray(axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1.0e-9:
        return 0.0, True
    axis = axis / axis_norm
    a = np.asarray(up_prev, dtype=float)
    b = np.asarray(up_curr, dtype=float)
    a = a - axis * float(np.dot(a, axis))
    b = b - axis * float(np.dot(b, axis))
    if float(np.linalg.norm(a)) < PHASE_EPS or float(np.linalg.norm(b)) < PHASE_EPS:
        return 0.0, False
    return math.atan2(float(np.dot(axis, np.cross(a, b))),
                       float(np.dot(a, b))), True


def select_grounded_reference(samples, first_air_t, lookback_s=0.5,
                              soft_tilt=math.radians(20.0),
                              hard_tilt=math.radians(30.0)):
    """Last grounded sample before lift, with a bounded tilt.

    ``samples`` is a chronological iterable of dicts containing ``t``, ``quat``
    and ``foot_force``.  The reference must still be in contact (foot force > 0)
    and within ``lookback_s`` of the first airborne sample.  Prefer the last
    sample below ``soft_tilt``; fall back to ``hard_tilt``; otherwise invalid.
    """
    candidates = []
    for sample in samples:
        t = float(sample["t"])
        if t < first_air_t - lookback_s or t > first_air_t + 1.0e-9:
            continue
        if sum(int(v) for v in sample.get("foot_force", ())) <= 0:
            continue
        if "quat" not in sample:
            continue
        candidates.append((body_tilt(sample["quat"]), sample))
    if not candidates:
        return None, False
    soft = [item for item in candidates if item[0] <= soft_tilt]
    if soft:
        return soft[-1][1], True
    hard = [item for item in candidates if item[0] <= hard_tilt]
    if hard:
        return hard[-1][1], True
    return None, False


def stand_metrics(samples, nominal=STAND_NOMINAL, tail_s=STAND_TAIL_S,
                  min_foot_force=1.0, foot_fraction=0.9):
    """Score a stand over a stable tail of at least ``tail_s`` seconds.

    Pose success uses the worst per-leg/per-sample joint error, not the mean
    over legs, so a single bad or asymmetric leg cannot hide behind three good
    legs.  The mean pose error is still reported for diagnostics.  All four feet
    must be loaded for at least ``foot_fraction`` of the tail.
    """
    failures = []
    if not samples:
        return {"success": False, "failures": ["no_samples"],
                "tail_s": tail_s, "tail_duration_s": 0.0, "window_s": 0.0,
                "body_height_mean": None, "body_height_center_mean": None,
                "max_tilt": None, "foot_radius": FOOT_RADIUS,
                "pose_error_hip_thigh_calf": [None, None, None],
                "pose_error_worst_hip_thigh_calf": [None, None, None],
                "mirror_error_worst": [None] * 6,
                "dq_rms": None, "q_mean": None,
                "foot_force_min": [None] * 4,
                "foot_force_loaded_fraction": [None] * 4,
                "height_range": None, "tilt_range": None, "stable": False,
                "pose_ok": False, "mirror_ok": False, "feet_ok": False,
                "enough_tail": False}

    last_t = float(samples[-1]["t"])
    first_t = float(samples[0]["t"])
    window_s = last_t - first_t
    tail = [s for s in samples if float(s["t"]) >= last_t - tail_s]
    tail_duration = float(tail[-1]["t"]) - float(tail[0]["t"]) if tail else 0.0
    enough_tail = bool(window_s >= tail_s - 1.0e-6
                       and tail_duration >= tail_s - 0.05)
    if not enough_tail:
        failures.append("insufficient_stand_tail")

    q = np.asarray([s["q"] for s in tail], dtype=float)          # (N, 12)
    relative = q.reshape(-1, 4, 3) - np.asarray(nominal, dtype=float)
    q_error = np.abs(relative)
    q_error_mean = q_error.mean(axis=(0, 1))
    q_error_worst = q_error.max(axis=(0, 1))
    pose_ok = bool(np.all(q_error_worst <= STAND_POSE_TOL))
    if not pose_ok:
        failures.append("pose_all_joints")
    # Training's left_right_mirror_error, in hardware order FR, FL, RR, RL.
    mirror = np.stack((
        relative[:, 1, 0] + relative[:, 0, 0],   # FL + FR hip
        relative[:, 3, 0] + relative[:, 2, 0],   # RL + RR hip
        relative[:, 1, 1] - relative[:, 0, 1],   # FL - FR thigh
        relative[:, 3, 1] - relative[:, 2, 1],   # RL - RR thigh
        relative[:, 1, 2] - relative[:, 0, 2],   # FL - FR calf
        relative[:, 3, 2] - relative[:, 2, 2],   # RL - RR calf
    ), axis=1)
    mirror_worst = np.abs(mirror).max(axis=0)
    mirror_ok = bool(np.all(mirror_worst <= STAND_MIRROR_TOL))
    if not mirror_ok:
        failures.append("pose_mirror")

    forces = np.asarray([s["foot_force"] for s in tail], dtype=float)  # (N, 4)
    loaded_fraction = (forces >= min_foot_force).mean(axis=0)
    feet_ok = bool(np.all(loaded_fraction >= foot_fraction))
    if not feet_ok:
        failures.append("foot_contact")

    dq = np.asarray([s["dq"] for s in tail], dtype=float)
    dq_rms = float(np.sqrt(np.mean(np.square(dq))))
    if dq_rms >= 1.0:
        failures.append("joint_velocity")

    heights_diag = np.asarray([s["body_height"] for s in tail], dtype=float)
    heights = np.asarray([
        s.get("body_height_center", s["body_height"] - FOOT_RADIUS)
        for s in tail
    ], dtype=float)
    tilts = np.asarray([body_tilt(s["quat"]) for s in tail], dtype=float)
    height = float(np.mean(heights))
    height_diag = float(np.mean(heights_diag))
    tilt = float(np.max(tilts))
    height_range = float(np.max(heights) - np.min(heights))
    tilt_range = float(np.max(tilts) - np.min(tilts))
    stable = bool(height_range <= .08 and tilt_range <= .15)
    if not stable:
        failures.append("tail_not_stable")
    if height < .2:
        failures.append("height_center")
    if tilt >= math.pi / 6.0:
        failures.append("tilt")

    success = bool(enough_tail and height >= .2 and tilt < math.pi / 6.0
                   and pose_ok and mirror_ok and feet_ok and dq_rms < 1.0
                   and stable)
    return {
        "success": success, "failures": failures,
        "tail_s": tail_s, "tail_duration_s": tail_duration, "window_s": window_s,
        "enough_tail": enough_tail, "stable": stable,
        # Diagnostic height above the reported foot bottom, kept for baseline
        # comparison.  body_height_center_mean is the scored training height.
        "body_height_mean": height_diag,
        "body_height_center_mean": height, "max_tilt": tilt,
        "foot_radius": FOOT_RADIUS,
        "height_range": height_range, "tilt_range": tilt_range,
        "pose_error_hip_thigh_calf": q_error_mean.tolist(),
        "pose_error_worst_hip_thigh_calf": q_error_worst.tolist(),
        "mirror_error_worst": mirror_worst.tolist(),
        "dq_rms": dq_rms, "q_mean": q.mean(axis=0).tolist(),
        "foot_force_min": forces.min(axis=0).tolist(),
        "foot_force_loaded_fraction": loaded_fraction.tolist(),
        "pose_ok": pose_ok, "mirror_ok": mirror_ok, "feet_ok": feet_ok,
    }


def summarize_chain(flights, recovery):
    """Target is the first intentional target, never a retry or recovery hop.

    ``success`` is the target result only.  ``chain_success`` additionally
    requires the recovery hop and the recovered stand.  A failed target cannot
    be masked by a later retry because only the first ``role == "target"``
    flight is considered.
    """
    targets = [f for f in flights if f.get("role") == "target"]
    recovery_hops = [f for f in flights if f.get("role") == "recovery_hop"]
    target = targets[0] if targets else None
    target_ok = bool(target and target.get("success"))
    recovery_hop_ok = bool(recovery_hops and recovery_hops[0].get("success"))
    stand_ok = bool(recovery and recovery.get("success"))
    # Energy-building retries are allowed, but the entry flight immediately
    # preceding the target must itself have landed successfully.
    target_index = next((i for i, f in enumerate(flights) if f.get("role") == "target"), len(flights))
    entries = [f for f in flights[:target_index] if f.get("role") == "entry"]
    entry_ok = bool(not entries or entries[-1].get("success"))
    return {
        "success": target_ok,
        "target_success": target_ok,
        "recovery_hop_success": recovery_hop_ok,
        "recovery_stand_success": stand_ok,
        "entry_success": entry_ok,
        "chain_success": bool(entry_ok and target_ok and recovery_hop_ok and stand_ok),
        "target_takeoff_t": target.get("takeoff_t") if target else None,
    }


def command_switch_ready(phase, t, flight, velocity_z, new_contact):
    """Whether the next command should be sent at this flight sample.

    ``apex`` (default) and ``landing`` keep the manual-like timing: after the
    apex, or on touchdown.  ``takeoff`` fires once the flight has been airborne
    for ``TAKEOFF_SWITCH_DELAY_S``, i.e. the first sample after liftoff, which is
    the training reveal (``reveal_lead_time`` covers the whole flight).  The
    caller's ``pending`` guard makes any phase fire at most once per flight.
    """
    if phase == "landing":
        return bool(new_contact)
    if phase == "takeoff":
        return float(t) >= float(flight["first_air_t"]) + TAKEOFF_SWITCH_DELAY_S
    return bool(velocity_z <= 0 and t - flight["takeoff_t"] > .1)


def result_counts(records):
    counts = {}
    for record in records:
        row = counts.setdefault(record["skill"], {
            "attempts": 0, "setup_errors": 0, "evaluated": 0,
            "successes": 0, "chain_successes": 0, "startup_stand_successes": 0,
        })
        row["attempts"] += 1
        if "error_phase" in record:
            row["setup_errors"] += 1
            continue
        row["evaluated"] += 1
        row["successes"] += int(bool(record.get("success")))
        row["chain_successes"] += int(bool(record.get("chain_success")))
        row["startup_stand_successes"] += int(bool(record["stand"]["success"]))
    return counts


class TrampolineNode(EvaluationNode):
    def __init__(self, args):
        super().__init__(50., args.status_timeout, "trampoline_sim2sim_evaluator")
        self.low = None
        self.lowcmd = None
        self.sample = None
        self.state_received_at = None
        self.trace = None
        self.phase = "setup"
        self.tau_peak = np.zeros(MOTOR_COUNT)
        self.tau_samples = 0
        self.low_sub = self.create_subscription(LowState, "/lowstate", self.on_low, 1)
        self.cmd_sub = self.create_subscription(LowCmd, "/lowcmd", self.on_cmd, 1)
        self.sport_sub = self.create_subscription(SportModeState, "/sportmodestate", self.on_sport, 1)

    def reset_torque_peak(self):
        self.tau_peak = np.zeros(MOTOR_COUNT)
        self.tau_samples = 0

    def torque_peak(self):
        """Peak |tau_est| over every LowState, not only 50 Hz trace samples.

        tau_est comes from the MuJoCo ``jointactuatorfrc`` sensor (noise 0.01),
        i.e. the actual actuator force after the simulator's ctrl/force limit.
        """
        return {
            "source": "LowState.motor_state[].tau_est (jointactuatorfrc, noise 0.01)",
            "samples": self.tau_samples,
            "abs_max": float(self.tau_peak.max()) if self.tau_samples else None,
            "abs_max_by_motor": self.tau_peak.tolist(),
        }

    def on_low(self, msg):
        self.low = msg
        tau = np.asarray([m.tau_est for m in msg.motor_state[:MOTOR_COUNT]], dtype=float)
        if tau.size == MOTOR_COUNT:
            self.tau_peak = np.maximum(self.tau_peak, np.abs(tau))
            self.tau_samples += 1

    def on_cmd(self, msg):
        self.lowcmd = msg

    def on_sport(self, msg):
        if self.low is None:
            return
        self.state_received_at = time.monotonic()
        low = self.low
        feet = np.asarray(msg.foot_position_body).reshape(4, 3)
        # The simulator's FR_pos/... sensors are WORLD coordinates despite the ROS field name.
        self.sample = {
            "t": msg.stamp.sec + msg.stamp.nanosec * 1e-9,
            "position": [float(x) for x in msg.position], "velocity": [float(x) for x in msg.velocity],
            "quat": [float(x) for x in low.imu_state.quaternion], "omega": [float(x) for x in low.imu_state.gyroscope],
            "q": [m.q for m in low.motor_state[:12]],
            "dq": [m.dq for m in low.motor_state[:12]],
            "tau_est": [m.tau_est for m in low.motor_state[:12]],
            "foot_force": [int(x) for x in low.foot_force],
            # Bridge reports foot bottoms (world z minus the foot radius);
            # keep it for baseline comparison and add the training-compatible
            # height above foot centres.
            "body_height": float(msg.position[2] - np.mean(feet[:, 2])),
            "body_height_center": float(msg.position[2] - np.mean(feet[:, 2])
                                        - FOOT_RADIUS),
            "keys": int(self.command.keys), "phase": self.phase,
        }
        if self.lowcmd is not None:
            self.sample["q_target"] = [m.q for m in self.lowcmd.motor_cmd[:12]]
            self.sample["tau_command"] = [m.tau + m.kp*(m.q-low.motor_state[i].q)
                + m.kd*(m.dq-low.motor_state[i].dq)
                for i, m in enumerate(self.lowcmd.motor_cmd[:12])]
        if self.trace:
            self.trace.write(json.dumps(self.sample) + "\n")

    def extra_freshness(self):
        return {"robot state": self.state_received_at}

    def preflight_extra_ready(self):
        return self.sample is not None


class TrampolineEvaluator(ControllerDriver):
    def wait_sim(self, duration):
        start = self.node.sample["t"]
        deadline = time.monotonic() + duration * 3 + 5
        while self.node.sample["t"] - start < duration:
            self.node.spin_once(.002)
            self.node.assert_fresh()
            if time.monotonic() > deadline:
                raise EvaluationError("simulation clock stopped advancing")

    def reset(self):
        """Reset, wait for fresh ready telemetry, then start and arm the episode.

        A reset/setup failure is raised, never recorded as a skill success.
        """
        self.node.phase = "reset"
        self.normalize_to_fixed_stand()
        previous_episode_id = self.node.episode_status.episode_id
        response = self.node.call_reset(self.args.service_timeout,
                                       base_x_offset=0., base_y_offset=0., base_yaw_offset=0.)
        if response.episode_id <= previous_episode_id:
            raise EvaluationError(
                "ResetEpisode did not advance episode_id "
                f"({previous_episode_id} -> {response.episode_id})")
        reset_stamp_ns = stamp_to_ns(response.reset_stamp)
        reset_received_at = time.monotonic()

        def reset_is_ready():
            episode = self.node.episode_status
            deploy = self.node.deploy_status
            age = time.monotonic() - reset_received_at
            return (
                episode.episode_id == response.episode_id
                and episode.outcome == EpisodeStatus.WAITING
                and stamp_to_ns(episode.stamp) > reset_stamp_ns
                and abs(episode.roll) <= self.args.max_settle_tilt
                and abs(episode.pitch) <= self.args.max_settle_tilt
                and episode.angular_velocity_rad_s <= self.args.max_settle_angular_velocity
                and episode.linear_velocity_m_s <= self.args.max_settle_linear_velocity
                and deploy.safety_ok
                and deploy.controller_state == DeployStatus.CONTROLLER_STATE_FIX_STAND
                and deploy.fix_stand_ready
                and age >= self.args.settle_time
            )

        try:
            self.node.wait_until(
                reset_is_ready, timeout=self.args.reset_ready_timeout,
                description="fresh reset episode ready and settled",
                fail_on_estop=True)
        except SafetyStop as error:
            raise TrialFailure("DEPLOY_ESTOP", str(error)) from error
        except TransitionTimeout as error:
            raise TrialFailure("RESET_READY_TIMEOUT", str(error)) from error

        self.node.phase = "stand"
        self.enter_policy()
        self.arm_episode(response.episode_id)

    def stand(self):
        samples = []
        start = self.node.sample["t"]
        last_t = -1
        deadline = time.monotonic() + self.args.stand_time * 3 + 5
        while self.node.sample["t"] - start < self.args.stand_time:
            self.node.spin_once(.002)
            self.node.assert_fresh()
            if time.monotonic() > deadline:
                raise EvaluationError("simulation clock stopped during stand")
            s = self.node.sample
            if s["t"] == last_t:
                continue
            last_t = s["t"]
            samples.append(s.copy())
        result = stand_metrics(samples, STAND_NOMINAL, tail_s=STAND_TAIL_S)
        result["stand_time_s"] = self.args.stand_time
        result["torque"] = self.node.torque_peak()
        return result

    @staticmethod
    def _first_air_time(history, contact_since, fallback_t):
        """First force-zero sample while rising, used only to bound the anchor search."""
        for sample in history:
            if (float(sample["t"]) > contact_since
                    and sum(int(v) for v in sample["foot_force"]) == 0
                    and float(sample["velocity"][2]) > .4):
                return float(sample["t"])
        return fallback_t

    def _flight_reference(self, history, first_air_t, name, takeoff_R, takeoff_tilt):
        """Freeze grounded yaw/axis, then anchored pre-rotation or anchored increment.

        Returns a dict with the reference frame and the initial angle.  When the
        reference is invalid the caller must not score a rotation success.
        """
        anchor, valid = select_grounded_reference(history, first_air_t)
        if anchor is None:
            valid = False
            anchor_R = takeoff_R
            anchor_t = float(first_air_t)
            anchor_position = list(history[-1]["position"]) if history else [0., 0., 0.]
            anchor_tilt = takeoff_tilt
            yaw0 = None
        else:
            anchor_R = rotation(anchor["quat"])
            anchor_t = float(anchor["t"])
            anchor_position = list(anchor["position"])
            anchor_tilt = body_tilt(anchor["quat"])
            yaw0 = yaw_twist(anchor["quat"])
        yaw_valid = yaw0 is not None
        if yaw0 is None:
            yaw0 = math.atan2(float(anchor_R[1, 0]), float(anchor_R[0, 0]))
        axis_body = np.asarray(SKILLS[name][1], dtype=float)
        if np.linalg.norm(axis_body) > 0.0 and valid:
            cy, sy = math.cos(yaw0), math.sin(yaw0)
            axis = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]]) @ axis_body
        else:
            axis = axis_body

        if np.linalg.norm(axis) == 0.0:
            return {"reference_valid": valid, "reference_mode": "hop",
                    "reference_t": anchor_t, "reference_tilt": anchor_tilt,
                    "reference_position": anchor_position, "yaw0": yaw0,
                    "yaw_valid": yaw_valid, "axis": axis.tolist(),
                    "angle": 0.0, "pre_rotation": 0.0, "pre_rotation_abs": 0.0,
                    "phase_valid": True, "last_R": takeoff_R}

        pre, pre_valid = pre_rotation_about_axis(takeoff_R[:, 2], axis)
        mode = "anchored_pre"
        angle = pre
        phase_valid = True
        if not valid:
            mode = "invalid"
            angle = 0.0
            phase_valid = False
        elif (not pre_valid) or takeoff_tilt > math.radians(120.0):
            # The late takeoff is already too far around for the absolute
            # pre-rotation to be unambiguous; integrate from the grounded anchor.
            angle, ok = pre_rotation_about_axis(anchor_R[:, 2], axis)
            previous_up = anchor_R[:, 2]
            previous_time = anchor_t
            for sample in history:
                if sample["t"] <= anchor_t:
                    continue
                up = rotation(sample["quat"])[:, 2]
                increment, valid_increment = phase_increment(previous_up, up, axis)
                if (not valid_increment or sample["t"] - previous_time > .05
                        or abs(increment) > math.pi / 2):
                    ok = False
                    break
                angle += increment
                previous_up = up
                previous_time = sample["t"]
            if ok:
                mode = "anchored_increment"
            else:
                mode = "invalid"
                angle = 0.0
                phase_valid = False
        return {"reference_valid": bool(valid and phase_valid),
                "reference_mode": mode, "reference_t": anchor_t,
                "reference_tilt": anchor_tilt, "reference_position": anchor_position,
                "yaw0": yaw0, "yaw_valid": yaw_valid,
                "axis": axis.tolist(), "angle": angle,
                "pre_rotation": angle, "pre_rotation_abs": pre,
                "phase_valid": phase_valid, "last_R": takeoff_R}

    def run_skill(self, skill):
        node = self.node
        node.reset_torque_peak()
        node.phase = "hop"
        node.set_command(keys=KEY_A)
        release_at = node.sample["t"] + self.args.button_hold
        start = node.sample["t"]
        wall_deadline = time.monotonic() + self.args.skill_timeout * 3 + 5
        previous_t = start
        contact = True
        filtered_force = sum(node.sample["foot_force"])
        contact_since = start
        flight = None
        flights = []
        requested = "hop"
        # A hop trial's first hop is the target; a flip trial's first hop is the entry.
        next_role = "target" if skill == "hop" else "entry"
        pending = False
        target_done = False
        chain_terminated = False
        recovery = None
        last_R = None
        history = deque(maxlen=600)
        phase_last_t = start
        phase_skips = 0

        while node.sample["t"] - start < self.args.skill_timeout:
            node.spin_once(.001)
            node.assert_fresh()
            if time.monotonic() > wall_deadline:
                raise EvaluationError("simulation clock stopped")
            s = node.sample
            t = s["t"]
            if t <= previous_t:
                continue
            dt = t - previous_t
            previous_t = t
            history.append({"t": t, "quat": list(s["quat"]),
                            "position": list(s["position"]),
                            "foot_force": list(s["foot_force"]),
                            "velocity": list(s["velocity"])})
            R = rotation(s["quat"])
            filtered_force += (1-math.exp(-dt/.025)) * (sum(s["foot_force"]) - filtered_force)
            new_contact = filtered_force > (10 if contact else 30)
            if t >= release_at:
                node.set_command()
                release_at = math.inf
            if (requested != "stand" and contact and not new_contact
                    and t-contact_since > .05 and s["velocity"][2] > .4):
                first_air_t = self._first_air_time(history, contact_since, t)
                takeoff_tilt = body_tilt(s["quat"])
                ref = self._flight_reference(history, first_air_t, requested, R, takeoff_tilt)
                flight = {
                    "skill": requested, "role": next_role,
                    "takeoff_t": ref["reference_t"] if ref["reference_valid"] else t,
                    "detected_takeoff_t": t, "first_air_t": first_air_t,
                    "reference_t": ref["reference_t"],
                    "reference_tilt": ref["reference_tilt"],
                    "reference_mode": ref["reference_mode"],
                    "reference_valid": ref["reference_valid"],
                    "position0": ref["reference_position"],
                    "apex": s["position"][2],
                    "angle": ref["angle"],
                    "pre_rotation": ref["pre_rotation"],
                    "pre_rotation_abs": ref["pre_rotation_abs"],
                    "yaw0": ref["yaw0"], "yaw_valid": ref["yaw_valid"],
                    "axis": ref["axis"], "phase_valid": ref["phase_valid"],
                    "phase_skips": 0,
                }
                last_R = ref["last_R"]
                phase_last_t = t
            if flight:
                flight["apex"] = max(flight["apex"], s["position"][2])
                axis = np.asarray(flight["axis"], dtype=float)
                if np.linalg.norm(axis) > 0.0:
                    increment, ok = phase_increment(last_R[:, 2], R[:, 2], axis)
                    if ok:
                        flight["angle"] += increment
                        last_R = R
                        phase_last_t = t
                    else:
                        flight["phase_skips"] += 1
                        phase_skips += 1
                        if t - phase_last_t > 0.2:
                            flight["phase_valid"] = False
                else:
                    last_R = R

                # Send the next command at apex (manual-like), 20 ms after the
                # first airborne sample (training reveal), or on touchdown.
                ready = command_switch_ready(
                    self.args.switch_phase, t, flight, s["velocity"][2],
                    new_contact)
                if ready and not pending:
                    role = flight["role"]
                    if chain_terminated:
                        pending = True
                    elif role == "target" and not target_done:
                        if self.args.full_chain:
                            requested = "hop"
                            next_role = "recovery_hop"
                            node.set_command(keys=SKILLS["hop"][0])
                        else:
                            chain_terminated = True
                            pending = True
                            requested = skill
                            next_role = None
                    elif role == "entry":
                        if (self.args.require_energy_height
                                and abs(flight["apex"] - self.args.height) > .1):
                            requested = "hop"
                            next_role = "entry"
                            node.set_command(keys=SKILLS["hop"][0])
                        elif skill != "hop":
                            requested = skill
                            next_role = "target"
                            node.set_command(keys=SKILLS[skill][0])
                        else:
                            requested = "hop"
                            next_role = "target"
                            node.set_command(keys=SKILLS["hop"][0])
                    elif role == "recovery_hop":
                        requested = "stand"
                        next_role = None
                        node.set_command(keys=KEY_B)
                    else:
                        requested = "hop"
                        next_role = "entry"
                        node.set_command(keys=SKILLS["hop"][0])
                    if not chain_terminated:
                        node.phase = requested
                        release_at = t + self.args.button_hold
                        pending = True

                if new_contact and not contact:
                    flight["landing_t"] = t
                    flight["tilt"] = math.acos(float(np.clip(R[2, 2], -1, 1)))
                    flight["displacement"] = float(np.linalg.norm(
                        np.array(s["position"][:2]) - np.array(flight["position0"][:2])))
                    turns = SKILLS[flight["skill"]][2]
                    yaw1 = yaw_twist(s["quat"])
                    if yaw1 is None or not flight["yaw_valid"]:
                        flight["yaw_valid"] = False
                        flight["yaw_error"] = None
                        yaw_ok = False
                    else:
                        flight["yaw_error"] = wrap_to_pi(yaw1 - flight["yaw0"])
                        yaw_ok = abs(flight["yaw_error"]) <= math.pi / 6.0
                    angle_ok = abs(flight["angle"] - turns*2*math.pi) <= math.pi/4
                    flight["rotation_landing_ok"] = bool(
                        flight["reference_valid"] and flight["phase_valid"]
                        and yaw_ok and angle_ok
                        and flight["tilt"] <= math.pi/6
                        and flight["displacement"] < .75)
                    flight["height_ok"] = abs(flight["apex"]-self.args.height) <= (.2 if turns == 2 else .1)
                    flight["success"] = bool(flight["rotation_landing_ok"] and flight["height_ok"])
                    if not flight["reference_valid"]:
                        flight["failure"] = "invalid_takeoff_reference"
                    elif not flight["phase_valid"]:
                        flight["failure"] = "invalid_phase_increment"
                    elif not yaw_ok:
                        flight["failure"] = "landing_yaw"
                    elif not angle_ok:
                        flight["failure"] = "rotation_angle"
                    elif flight["tilt"] > math.pi/6:
                        flight["failure"] = "landing_tilt"
                    elif flight["displacement"] >= .75:
                        flight["failure"] = "landing_displacement"
                    elif not flight["height_ok"]:
                        flight["failure"] = "apex_height"
                    landed_role = flight["role"]
                    flights.append(flight)
                    if landed_role == "target":
                        target_done = True
                    recovered_hop = landed_role == "recovery_hop"
                    flight = None
                    pending = False
                    if target_done and not self.args.full_chain:
                        self.wait_sim(.3)
                        break
                    if target_done and self.args.full_chain and next_role == "target":
                        # The apex switch was missed (very short flight).  Send
                        # the recovery hop at touchdown so the full chain still
                        # tests hop + stand instead of leaving the command stale.
                        requested = "hop"
                        next_role = "recovery_hop"
                        node.phase = "hop"
                        node.set_command(keys=SKILLS["hop"][0])
                        release_at = t + self.args.button_hold
                        pending = True
                    if recovered_hop:
                        if next_role is not None or requested != "stand":
                            # The recovery-hop apex switch was missed; command B
                            # at touchdown instead of scoring a hop as a stand.
                            requested = "stand"
                            next_role = None
                            node.phase = "stand"
                            node.set_command(keys=KEY_B)
                            release_at = t + self.args.button_hold
                            pending = True
                        node.phase = "final_stand"
                        recovery = self.stand()
                        break
            if new_contact and not contact:
                contact_since = t
            contact = new_contact
            if not node.deploy_status.safety_ok:
                break

        if flight:
            flight["success"] = False
            flight["failure"] = "no_landing_before_timeout"
            flights.append(flight)
        summary = summarize_chain(flights, recovery)
        summary.update({
            "flights": flights,
            "recovery_stand": recovery,
            "phase_skips": phase_skips,
            "safety_ok": bool(node.deploy_status.safety_ok),
            "final_position": node.sample["position"],
            "final_q": node.sample["q"],
            "torque": node.torque_peak(),
        })
        return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--skills", nargs="+", choices=["stand", *SKILLS], default=["stand", *SKILLS])
    p.add_argument("--trials", type=positive_int, default=1)
    p.add_argument("--switch-phase", choices=["apex", "takeoff", "landing"],
                   default="apex",
                   help="when to send the next command in a flight: apex "
                        "(default), takeoff (training reveal, 20 ms after the "
                        "first airborne sample), or landing")
    p.add_argument("--full-chain", action="store_true", help="also evaluate a recovery hop and B stand")
    p.add_argument("--require-energy-height", action="store_true",
                   help="retry the initial hop until its apex is within 0.1 m, like training")
    p.add_argument("--height", type=positive_float, default=1.3)
    p.add_argument("--stand-time", type=positive_float, default=3.)
    p.add_argument("--settle-time", type=positive_float, default=3.)
    p.add_argument("--reset-ready-timeout", type=positive_float, default=10.0)
    p.add_argument("--max-settle-tilt", type=positive_float, default=0.15)
    p.add_argument("--max-settle-angular-velocity", type=positive_float, default=0.30)
    p.add_argument("--max-settle-linear-velocity", type=positive_float, default=0.05)
    p.add_argument("--skill-timeout", type=positive_float, default=10.)
    p.add_argument("--button-hold", type=positive_float, default=.06)
    p.add_argument("--status-timeout", type=positive_float, default=2.)
    p.add_argument("--service-timeout", type=positive_float, default=5.)
    p.add_argument("--transition-timeout", type=positive_float, default=8.)
    args = p.parse_args()
    domain = validate_simulation_environment()
    args.output.mkdir(parents=True, exist_ok=False)
    rclpy.init(args=[])
    node = TrampolineNode(args)
    evaluator = TrampolineEvaluator(node, args)
    records = []
    error = None
    try:
        node.wait_for_preflight(20.)
        node.publishing_enabled = True
        for trial in range(args.trials):
            for skill in args.skills:
                phase = "reset"
                try:
                    with (args.output / f"{trial:03d}-{skill}.trace.jsonl").open("w") as trace:
                        node.trace = trace
                        evaluator.reset()
                        phase = "stand"
                        node.reset_torque_peak()
                        stand = evaluator.stand()
                        if skill == "stand":
                            result = stand
                        else:
                            phase = "skill"
                            result = evaluator.run_skill(skill)
                        record = {"trial": trial, "skill": skill, "stand": stand, **result}
                        if "chain_success" in record:
                            record["chain_success"] = bool(record["chain_success"] and stand["success"])
                        records.append(record)
                        print(json.dumps(record), flush=True)
                        (args.output / "results.json").write_text(json.dumps(records, indent=2)+"\n")
                except (EvaluationError, TrialFailure) as exc:
                    # A reset/startup failure is not a skill result.  Record it
                    # explicitly and try to recover the controller for the next
                    # skill; never attach success/chain_success to it.
                    record = {"trial": trial, "skill": skill,
                              "error_phase": phase, "error": str(exc)}
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    (args.output / "results.json").write_text(json.dumps(records, indent=2)+"\n")
                    try:
                        evaluator.normalize_to_fixed_stand()
                    except Exception as recover_exc:
                        error = f"recovery after {skill} failed: {recover_exc}"
                        raise
                    continue
                finally:
                    node.trace = None
                evaluator.pulse(KEY_SELECT)
    except (Exception, KeyboardInterrupt) as exc:
        error = error or str(exc) or "interrupted"
        raise
    finally:
        node.trace = None
        if node.publishing_enabled and rclpy.ok():
            evaluator.pulse(KEY_SELECT)
            node.set_command()
        summary = {"ros_domain": domain, "arguments": {**vars(args), "output": str(args.output)},
                   "records": records, "counts": result_counts(records), "error": error}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        node.destroy_node()
        rclpy.shutdown()

    if any("error_phase" in record for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
