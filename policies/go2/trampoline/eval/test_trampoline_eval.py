"""Focused measurement tests for the trampoline sim2sim evaluator.

Run under the ROS environment used by the evaluator:

  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  python3 -m pytest src/legged_rl_deploy/policies/go2/trampoline/eval/test_trampoline_eval.py -q

These tests are synthetic and do not start ROS nodes or the simulator.  They pin
the measurement contract that fixes the back-2 reference-frame bug: grounded
heading/axis, anchored pre-rotation, projected-up phase increments, signed angle
(no abs), robust swing-twist yaw, reference validity, stand worst-leg/foot force,
and first-intentional-target chain scoring.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_trampoline_sim2sim as ev  # noqa: E402


def quat_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    return np.array([math.cos(half), *(math.sin(half) * axis)])


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def quat_yaw(yaw):
    return quat_axis_angle([0., 0., 1.], yaw)


def rotation_axis_from_yaw(yaw, body_axis):
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]]) @ np.asarray(body_axis, float)


def make_flip_history(yaw0, body_axis, takeoff_deg, end_deg, step_deg=1.0,
                      grounded_until_deg=None, reverse=False):
    """Chronological samples for a pure flip about a yawed horizontal axis."""
    world_axis = rotation_axis_from_yaw(yaw0, body_axis)
    if reverse:
        world_axis = -world_axis
    q_anchor = quat_yaw(yaw0)
    if grounded_until_deg is None:
        grounded_until_deg = takeoff_deg
    samples = []
    angle = 0.0
    while angle <= end_deg + 1e-9:
        theta = math.radians(angle)
        q = quat_mul(quat_axis_angle(world_axis, theta), q_anchor)
        samples.append({
            "t": len(samples) * 0.005,
            "quat": q.tolist(),
            "position": [0., 0., 0.3 + 0.002 * len(samples)],
            "velocity": [0., 0., 3.0],
            "foot_force": [10, 10, 10, 10] if angle <= grounded_until_deg else [0, 0, 0, 0],
        })
        angle += step_deg
    return samples, world_axis, q_anchor


def anchored_flight_angle(history, first_air_t, name, takeoff_R, takeoff_tilt,
                          end_R):
    ref = ev.TrampolineEvaluator._flight_reference(
        None, [s for s in history if s["t"] <= first_air_t], first_air_t,
        name, takeoff_R, takeoff_tilt)
    angle = ref["angle"]
    axis = np.asarray(ref["axis"], float)
    last_R = ref["last_R"]
    # Walk the remaining samples and use the same projected-up increment.
    start_i = next(i for i, s in enumerate(history) if s["t"] >= first_air_t)
    for sample in history[start_i:]:
        R = ev.rotation(sample["quat"])
        inc, ok = ev.phase_increment(last_R[:, 2], R[:, 2], axis)
        if ok:
            angle += inc
            last_R = R
    # Include the final synthetic end orientation if it is beyond the last sample.
    if np.linalg.norm(axis) > 0:
        inc, ok = ev.phase_increment(last_R[:, 2], end_R[:, 2], axis)
        if ok:
            angle += inc
    return ref, angle


@pytest.mark.parametrize("name,body_axis", [("back-2", (0., -1., 0.)),
                                            ("front-2", (0., 1., 0.)),
                                            ("right-2", (1., 0., 0.)),
                                            ("left-2", (-1., 0., 0.))])
def test_full_double_rotation_above_90_late_takeoff(name, body_axis):
    yaw0 = -0.80  # like the back-2 trace
    takeoff_deg = 98.0
    history, _, _ = make_flip_history(yaw0, body_axis, takeoff_deg, 720.0)
    # 1 degree per sample: index == angle in degrees.
    takeoff = history[int(round(takeoff_deg))]
    first_air_t = takeoff["t"]
    R_takeoff = ev.rotation(takeoff["quat"])
    R_end = ev.rotation(history[-1]["quat"])
    ref, angle = anchored_flight_angle(
        history, first_air_t, name, R_takeoff, ev.body_tilt(takeoff["quat"]), R_end)
    assert ref["reference_valid"], ref
    assert ref["reference_mode"] in ("anchored_pre", "anchored_increment")
    # The trace-like >90 case: Euler heading flips for pitch flips, while the
    # swing-twist heading stays.  A roll flip keeps body-x horizontal.
    euler_takeoff = math.atan2(R_takeoff[1, 0], R_takeoff[0, 0])
    if name.startswith(("front", "back")):
        assert abs(ev.wrap_to_pi(euler_takeoff - yaw0)) > math.pi / 2
    assert abs(ev.wrap_to_pi(ev.yaw_twist(takeoff["quat"]) - yaw0)) < 1e-6
    # Two full turns in this synthetic sequence.
    assert abs(angle - 4 * math.pi) < 1e-6
    assert angle > 0
    assert ref["pre_rotation_abs"] > 0


def test_genuine_wrong_direction_stays_negative():
    yaw0 = -0.80
    takeoff_deg = 98.0
    # Commanded back-1 axis; the robot rotates the opposite way.
    history, _, _ = make_flip_history(yaw0, (0., -1., 0.), takeoff_deg, 360.0,
                                      reverse=True)
    takeoff = history[int(round(takeoff_deg))]
    R_takeoff = ev.rotation(takeoff["quat"])
    R_end = ev.rotation(history[-1]["quat"])
    ref, angle = anchored_flight_angle(
        history, takeoff["t"], "back-2", R_takeoff,
        ev.body_tilt(takeoff["quat"]), R_end)
    target = 2 * math.pi
    assert angle < 0
    assert abs(angle - target) > math.pi / 4
    # This must be a failure, not abs(angle) relabeling.
    assert not (abs(angle - target) <= math.pi / 4)


@pytest.mark.parametrize("takeoff_deg", [130.0, 226.0])
def test_pre_rotation_fallback_above_120_and_full_rotation(takeoff_deg):
    yaw0 = 0.4
    history, _, _ = make_flip_history(yaw0, (0., -1., 0.), takeoff_deg, 360.0,
                                      step_deg=2.0)
    takeoff = history[int(round(takeoff_deg / 2.0))]
    R_takeoff = ev.rotation(takeoff["quat"])
    R_end = ev.rotation(history[-1]["quat"])
    ref, angle = anchored_flight_angle(
        history, takeoff["t"], "back-2", R_takeoff,
        ev.body_tilt(takeoff["quat"]), R_end)
    assert ref["reference_valid"]
    assert ref["reference_mode"] == "anchored_increment"
    assert abs(angle - 2 * math.pi) < 1e-6
    assert angle > 0


def test_reference_invalid_without_grounded_sample():
    yaw0 = 0.2
    # Force is zero from the start, so no grounded anchor is available.
    history, _, _ = make_flip_history(yaw0, (0., -1., 0.), 0.0, 360.0,
                                      grounded_until_deg=-1.0)
    takeoff = history[0]
    ref = ev.TrampolineEvaluator._flight_reference(
        None, history, takeoff["t"], "back-2", ev.rotation(takeoff["quat"]),
        ev.body_tilt(takeoff["quat"]))
    assert not ref["reference_valid"]
    assert not ref["phase_valid"] or ref["reference_mode"] == "invalid"


def test_yaw_twist_stable_while_euler_flips():
    q = quat_axis_angle([0., 1., 0.], math.radians(100.0))
    R = ev.rotation(q)
    euler = math.atan2(R[1, 0], R[0, 0])
    assert abs(ev.wrap_to_pi(euler)) > math.pi / 2
    assert abs(ev.yaw_twist(q)) < 1e-6
    q180 = quat_axis_angle([0., 1., 0.], math.pi)
    assert ev.yaw_twist(q180) is None


def make_stand_samples(seconds=3.0, freq=200.0, q=None, foot_force=None,
                       body_height=0.25, dq=0.0):
    n = int(round(seconds * freq))
    q = ev.STAND_NOMINAL.copy() if q is None else np.asarray(q, float)
    if foot_force is None:
        foot_force = [10, 10, 10, 10]
    samples = []
    for i in range(n):
        samples.append({
            "t": i / freq,
            "q": q.reshape(-1).tolist(),
            "dq": [dq] * 12,
            "foot_force": list(foot_force),
            "body_height": body_height,
            "quat": [1., 0., 0., 0.],
        })
    return samples


def test_stand_metrics_stable_tail():
    samples = make_stand_samples()
    result = ev.stand_metrics(samples)
    assert result["success"], result
    assert result["enough_tail"]
    assert result["feet_ok"]
    assert result["pose_ok"]
    assert result["mirror_ok"]
    assert abs(result["tail_duration_s"] - 2.0) < 0.02
    assert result["pose_error_worst_hip_thigh_calf"] == [0., 0., 0.]
    # Diagnostic is above the foot bottom; the scored height is 22 mm lower.
    assert abs(result["body_height_mean"] - 0.25) < 1e-9
    assert abs(result["body_height_center_mean"]
               - (0.25 - ev.FOOT_RADIUS)) < 1e-9


def test_stand_worst_leg_catches_asymmetry_mean_hides():
    q = ev.STAND_NOMINAL.reshape(-1).copy()
    # Hardware order FR, FL, RR, RL.  FL thigh error 0.6; mean over legs is 0.15.
    q[4] = 1.5
    samples = make_stand_samples(q=q.reshape(4, 3))
    result = ev.stand_metrics(samples)
    assert not result["pose_ok"]
    assert not result["success"]
    assert result["pose_error_hip_thigh_calf"][1] < 0.4
    assert result["pose_error_worst_hip_thigh_calf"][1] > 0.3
    assert "pose_all_joints" in result["failures"]


def test_stand_foot_force_and_short_tail():
    samples = make_stand_samples()
    for s in samples:
        s["foot_force"] = [0, 10, 10, 10]
    result = ev.stand_metrics(samples)
    assert not result["feet_ok"]
    assert not result["success"]
    assert "foot_contact" in result["failures"]

    short = make_stand_samples(seconds=1.5)
    result = ev.stand_metrics(short)
    assert not result["enough_tail"]
    assert not result["success"]
    assert "insufficient_stand_tail" in result["failures"]


def test_switch_phase_fires_takeoff_apex_and_landing():
    # Flight established at t=1.00; the grounded anchor was taken at 0.95.
    flight = {"first_air_t": 1.00, "takeoff_t": .95}
    delay = ev.TAKEOFF_SWITCH_DELAY_S
    # takeoff: first sample at/after first_air + delay, even while rising.
    assert not ev.command_switch_ready(
        "takeoff", flight["first_air_t"], flight, 3.0, False)
    assert ev.command_switch_ready(
        "takeoff", flight["first_air_t"] + delay, flight, 3.0, False)
    assert ev.command_switch_ready(
        "takeoff", flight["first_air_t"] + .10, flight, 3.0, False)
    # apex keeps the manual-like timing: falling and >0.1 s after takeoff.
    assert not ev.command_switch_ready(
        "apex", flight["takeoff_t"] + .05, flight, -1.0, False)
    assert not ev.command_switch_ready(
        "apex", flight["takeoff_t"] + .12, flight, 1.0, False)
    assert ev.command_switch_ready(
        "apex", flight["takeoff_t"] + .12, flight, -.1, False)
    # landing fires only on a fresh contact, even while still rising.
    assert not ev.command_switch_ready(
        "landing", flight["first_air_t"] + .2, flight, 0.0, False)
    assert ev.command_switch_ready(
        "landing", flight["first_air_t"] + .2, flight, 0.0, True)


def test_chain_requires_target_recovery_hop_and_stand():
    target = {"role": "target", "success": True, "takeoff_t": 1.0}
    rec_hop = {"role": "recovery_hop", "success": True}
    good_stand = {"success": True}
    summary = ev.summarize_chain([target, rec_hop], good_stand)
    assert summary["success"] and summary["chain_success"]
    assert summary["target_success"] and summary["recovery_hop_success"]
    summary = ev.summarize_chain([{"role": "entry", "success": False}, target, rec_hop], good_stand)
    assert summary["target_success"] and not summary["chain_success"]

    summary = ev.summarize_chain([target, {"role": "recovery_hop", "success": False}],
                                 good_stand)
    assert summary["target_success"] and not summary["chain_success"]

    summary = ev.summarize_chain([{"role": "target", "success": False},
                                  {"role": "recovery_hop", "success": True}],
                                 good_stand)
    assert not summary["success"] and not summary["chain_success"]


def test_chain_uses_first_intentional_target_not_retry_or_recovery_hop():
    # A failed target followed by a successful target-like retry must not pass.
    flights = [
        {"role": "entry", "success": False},
        {"role": "target", "success": False},
        {"role": "target", "success": True},
        {"role": "recovery_hop", "success": True},
    ]
    summary = ev.summarize_chain(flights, {"success": True})
    assert not summary["target_success"]
    assert not summary["success"]
    assert not summary["chain_success"]

    # Hop case: the target is the first hop, not the recovery hop.
    hop_flights = [
        {"role": "target", "success": True, "takeoff_t": 1.0},
        {"role": "recovery_hop", "success": True, "takeoff_t": 2.0},
    ]
    summary = ev.summarize_chain(hop_flights, {"success": True})
    assert summary["target_takeoff_t"] == 1.0
    assert summary["chain_success"]


def test_stand_height_scores_foot_centres_but_keeps_diagnostic():
    # Bridge body_height = 0.21 is above the foot bottom; above foot centres it
    # is 0.188, below the training 0.2 m criterion.  The diagnostic must still
    # report 0.21 for baseline comparison.
    samples = make_stand_samples(body_height=0.21)
    result = ev.stand_metrics(samples)
    assert abs(result["body_height_mean"] - 0.21) < 1e-9
    assert abs(result["body_height_center_mean"] - 0.188) < 1e-9
    assert not result["success"]
    assert "height_center" in result["failures"]


def test_stand_mirror_pairs_are_checked_without_averaging():
    q = ev.STAND_NOMINAL.reshape(-1).copy()
    # Hardware FR hip index 0, FL hip index 3.  Both +0.15 relative; each hip
    # passes its own 0.2 rad bound, but the mirror sum is 0.30 > 0.2.
    q[0] = 0.25
    q[3] = 0.05
    samples = make_stand_samples(q=q.reshape(4, 3))
    result = ev.stand_metrics(samples)
    assert result["pose_ok"]
    assert not result["mirror_ok"]
    assert not result["success"]
    assert "pose_mirror" in result["failures"]
    assert result["mirror_error_worst"][0] > 0.2
