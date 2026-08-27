#!/usr/bin/env python3
"""Run repeated beam evaluations against the resident Unitree MuJoCo stack."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import rclpy
from legged_rl_deploy.msg import DeployStatus
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from unitree_go.msg import WirelessController
from unitree_mujoco.msg import EpisodeStatus
from unitree_mujoco.srv import ResetEpisode, StartEpisode


WIRELESS_TOPIC = "/wirelesscontroller"
DEPLOY_STATUS_TOPIC = "/legged_rl_deploy/status"
EPISODE_STATUS_TOPIC = "/unitree_mujoco/episode_status"
PROCESSED_DEPTH_TOPIC = "/unitree_go2_beam_depth/depth_m"
RESET_SERVICE = "/unitree_mujoco/reset_episode"
START_SERVICE = "/unitree_mujoco/start_episode"

KEY_START = 1 << 2
KEY_SELECT = 1 << 3
KEY_L2 = 1 << 5
KEY_A = 1 << 8
KEY_CLEAR_ESTOP = KEY_SELECT | KEY_START
KEY_FIX_STAND = KEY_L2 | KEY_A


class EvaluationError(RuntimeError):
    """A fatal setup, communication, or state-machine error."""


class TransitionTimeout(EvaluationError):
    """A controller transition did not finish before its deadline."""


class SafetyStop(EvaluationError):
    """Deploy reported that its safety flag is false."""


class TrialFailure(RuntimeError):
    """A recoverable failure attributed to the current trial."""

    def __init__(self, result: str, detail: str) -> None:
        super().__init__(detail)
        self.result = result
        self.detail = detail


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp_to_ns(stamp: object) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite value greater than zero")
    return value


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative value")
    return value


def velocity_value(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0.5 <= value <= 0.8:
        raise argparse.ArgumentTypeError("must be in [0.5, 0.8]")
    return value


def validate_simulation_environment() -> int:
    domain_text = os.environ.get("ROS_DOMAIN_ID")
    if domain_text is None:
        raise EvaluationError("ROS_DOMAIN_ID must be set explicitly to a nonzero value")
    try:
        domain_id = int(domain_text)
    except ValueError as error:
        raise EvaluationError(f"invalid ROS_DOMAIN_ID: {domain_text!r}") from error
    if not 1 <= domain_id <= 232:
        raise EvaluationError("ROS_DOMAIN_ID must be in [1, 232] for this simulation evaluator")

    rmw = os.environ.get("RMW_IMPLEMENTATION")
    if rmw != "rmw_cyclonedds_cpp":
        raise EvaluationError("RMW_IMPLEMENTATION must be rmw_cyclonedds_cpp")

    interface = os.environ.get("NetworkInterface")
    if interface != "lo":
        raise EvaluationError("NetworkInterface must be exactly 'lo'")

    cyclone_uri = os.environ.get("CYCLONEDDS_URI")
    if not cyclone_uri:
        raise EvaluationError("CYCLONEDDS_URI must explicitly bind CycloneDDS to 'lo'")
    try:
        root = ET.fromstring(cyclone_uri)
    except ET.ParseError as error:
        raise EvaluationError("CYCLONEDDS_URI must contain inline CycloneDDS XML") from error

    configured_interfaces = [
        element.attrib.get("name")
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "NetworkInterface"
    ]
    if not configured_interfaces or any(name != "lo" for name in configured_interfaces):
        raise EvaluationError(
            "every CYCLONEDDS_URI NetworkInterface must explicitly name 'lo'"
        )
    return domain_id


class EvaluationNode(Node):
    def __init__(self, heartbeat_hz: float, status_timeout: float) -> None:
        super().__init__("beam_sim2sim_evaluator")
        self.status_timeout = status_timeout
        self.deploy_status: DeployStatus | None = None
        self.episode_status: EpisodeStatus | None = None
        self.depth_stamp_ns: int | None = None
        self.deploy_received_at: float | None = None
        self.episode_received_at: float | None = None
        self.depth_received_at: float | None = None
        self.publishing_enabled = False
        self._last_graph_check = 0.0

        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.command_publisher = self.create_publisher(
            WirelessController, WIRELESS_TOPIC, command_qos
        )
        self.deploy_subscription = self.create_subscription(
            DeployStatus, DEPLOY_STATUS_TOPIC, self._on_deploy_status, reliable_qos
        )
        self.episode_subscription = self.create_subscription(
            EpisodeStatus, EPISODE_STATUS_TOPIC, self._on_episode_status, reliable_qos
        )
        self.depth_subscription = self.create_subscription(
            Image, PROCESSED_DEPTH_TOPIC, self._on_depth, sensor_qos
        )
        self.reset_client = self.create_client(ResetEpisode, RESET_SERVICE)
        self.start_client = self.create_client(StartEpisode, START_SERVICE)

        self.command = WirelessController()
        self.command_timer = self.create_timer(1.0 / heartbeat_hz, self._publish_command)

    def _on_deploy_status(self, message: DeployStatus) -> None:
        self.deploy_status = message
        self.deploy_received_at = time.monotonic()

    def _on_episode_status(self, message: EpisodeStatus) -> None:
        self.episode_status = message
        self.episode_received_at = time.monotonic()

    def _on_depth(self, message: Image) -> None:
        self.depth_stamp_ns = stamp_to_ns(message.header.stamp)
        self.depth_received_at = time.monotonic()

    def _publish_command(self) -> None:
        if self.publishing_enabled:
            self.command_publisher.publish(self.command)

    def set_command(self, *, keys: int = 0, ly: float = 0.0) -> None:
        self.command.lx = 0.0
        self.command.ly = float(ly)
        self.command.rx = 0.0
        self.command.ry = 0.0
        self.command.keys = int(keys)
        if self.publishing_enabled:
            self.command_publisher.publish(self.command)

    def spin_once(self, timeout: float = 0.02) -> None:
        rclpy.spin_once(self, timeout_sec=max(0.0, timeout))

    def pump(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            self.spin_once(min(0.02, deadline - time.monotonic()))

    def wait_for_preflight(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            self.spin_once(min(0.05, deadline - time.monotonic()))
            if (
                self.deploy_status is not None
                and self.episode_status is not None
                and self.depth_stamp_ns is not None
                and self.reset_client.service_is_ready()
                and self.start_client.service_is_ready()
            ):
                self.assert_fresh()
                self.assert_only_wireless_publisher(force=True)
                return
        missing = []
        if self.deploy_status is None:
            missing.append(DEPLOY_STATUS_TOPIC)
        if self.episode_status is None:
            missing.append(EPISODE_STATUS_TOPIC)
        if self.depth_stamp_ns is None:
            missing.append(PROCESSED_DEPTH_TOPIC)
        if not self.reset_client.service_is_ready():
            missing.append(RESET_SERVICE)
        if not self.start_client.service_is_ready():
            missing.append(START_SERVICE)
        raise EvaluationError("preflight timed out waiting for: " + ", ".join(missing))

    def assert_only_wireless_publisher(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_graph_check < 1.0:
            return
        self._last_graph_check = now
        publishers = self.get_publishers_info_by_topic(WIRELESS_TOPIC)
        if len(publishers) != 1:
            raise EvaluationError(
                f"expected only this evaluator to publish {WIRELESS_TOPIC}, "
                f"found {len(publishers)} publishers; start MuJoCo with --no-joystick"
            )

    def assert_fresh(self) -> None:
        now = time.monotonic()
        if (
            self.deploy_received_at is None
            or self.episode_received_at is None
            or self.depth_received_at is None
        ):
            raise EvaluationError("status or processed depth has not been received")
        deploy_age = now - self.deploy_received_at
        episode_age = now - self.episode_received_at
        depth_age = now - self.depth_received_at
        if deploy_age > self.status_timeout:
            raise EvaluationError(
                f"{DEPLOY_STATUS_TOPIC} is stale ({deploy_age:.3f} s)"
            )
        if episode_age > self.status_timeout:
            raise EvaluationError(
                f"{EPISODE_STATUS_TOPIC} is stale ({episode_age:.3f} s)"
            )
        if depth_age > self.status_timeout:
            raise EvaluationError(
                f"{PROCESSED_DEPTH_TOPIC} is stale ({depth_age:.3f} s)"
            )
        self.assert_only_wireless_publisher()

    def wait_until(
        self,
        predicate: Callable[[], bool],
        *,
        timeout: float,
        description: str,
        fail_on_estop: bool = False,
    ) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            self.spin_once(min(0.02, deadline - time.monotonic()))
            self.assert_fresh()
            if fail_on_estop and not self.deploy_status.safety_ok:
                raise SafetyStop("legged_rl_deploy entered e-stop")
            if predicate():
                return
        raise TransitionTimeout(f"timed out waiting for {description}")

    def call_reset(
        self,
        timeout: float,
        *,
        base_x_offset: float,
        base_y_offset: float,
        base_yaw_offset: float,
    ) -> ResetEpisode.Response:
        request = ResetEpisode.Request()
        request.base_x_offset = base_x_offset
        request.base_y_offset = base_y_offset
        request.base_yaw_offset = base_yaw_offset
        response = self._call_service(self.reset_client, request, timeout, RESET_SERVICE)
        if not response.success:
            raise EvaluationError(f"{RESET_SERVICE} rejected reset: {response.message}")
        return response

    def call_start(self, timeout: float) -> StartEpisode.Response:
        response = self._call_service(
            self.start_client, StartEpisode.Request(), timeout, START_SERVICE
        )
        if not response.success:
            raise EvaluationError(f"{START_SERVICE} rejected start: {response.message}")
        return response

    def _call_service(self, client: object, request: object, timeout: float, name: str):
        if not client.service_is_ready():
            raise EvaluationError(f"service disappeared before call: {name}")
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline and not future.done():
            self.spin_once(min(0.02, deadline - time.monotonic()))
        if not future.done():
            raise EvaluationError(f"service call timed out: {name}")
        error = future.exception()
        if error is not None:
            raise EvaluationError(f"service call failed: {name}: {error}") from error
        return future.result()


class BeamEvaluator:
    CONTROLLER_NAMES = {
        DeployStatus.CONTROLLER_STATE_IDLE: "IDLE",
        DeployStatus.CONTROLLER_STATE_FIX_STAND: "FIX_STAND",
        DeployStatus.CONTROLLER_STATE_PRE_IDLE: "PRE_IDLE",
        DeployStatus.CONTROLLER_STATE_HIGH_CONTROLLER: "HIGH_CONTROLLER",
    }
    OUTCOME_NAMES = {
        EpisodeStatus.WAITING: "WAITING",
        EpisodeStatus.RUNNING: "RUNNING",
        EpisodeStatus.SUCCESS: "SUCCESS",
        EpisodeStatus.SELF_COLLISION: "SELF_COLLISION",
        EpisodeStatus.ILLEGAL_CONTACT: "ILLEGAL_CONTACT",
        EpisodeStatus.FELL_OVER: "FELL_OVER",
        EpisodeStatus.TIMEOUT: "TIMEOUT",
    }
    TERMINAL_OUTCOMES = {
        EpisodeStatus.SUCCESS,
        EpisodeStatus.SELF_COLLISION,
        EpisodeStatus.ILLEGAL_CONTACT,
        EpisodeStatus.FELL_OVER,
        EpisodeStatus.TIMEOUT,
    }

    def __init__(self, node: EvaluationNode, args: argparse.Namespace) -> None:
        self.node = node
        self.args = args

    def pulse(self, keys: int) -> None:
        self.node.set_command(keys=keys, ly=0.0)
        self.node.pump(self.args.button_hold)
        self.node.set_command()
        self.node.pump(self.args.button_hold)

    def normalize_to_fixed_stand(self) -> None:
        self.node.set_command()
        deadline = time.monotonic() + 3.0 * self.args.transition_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            self.node.spin_once(0.02)
            self.node.assert_fresh()
            status = self.node.deploy_status

            if not status.safety_ok:
                self.node.get_logger().warning("clearing deploy e-stop with SELECT+START")
                self.pulse(KEY_CLEAR_ESTOP)
                try:
                    self.node.wait_until(
                        lambda: self.node.deploy_status.safety_ok
                        and self.node.deploy_status.controller_state
                        == DeployStatus.CONTROLLER_STATE_IDLE,
                        timeout=min(self.args.transition_timeout, deadline - time.monotonic()),
                        description="safe IDLE after SELECT+START",
                    )
                except TransitionTimeout:
                    continue
                continue

            state = status.controller_state
            if state == DeployStatus.CONTROLLER_STATE_FIX_STAND:
                try:
                    self.node.wait_until(
                        lambda: self.node.deploy_status.controller_state
                        == DeployStatus.CONTROLLER_STATE_FIX_STAND
                        and self.node.deploy_status.fix_stand_ready,
                        timeout=min(self.args.transition_timeout, deadline - time.monotonic()),
                        description="ready FIX_STAND",
                        fail_on_estop=True,
                    )
                    return
                except SafetyStop:
                    continue

            if state == DeployStatus.CONTROLLER_STATE_HIGH_CONTROLLER:
                self.node.get_logger().info("SELECT: HIGH_CONTROLLER -> FIX_STAND")
                self.pulse(KEY_SELECT)
                continue

            if state == DeployStatus.CONTROLLER_STATE_PRE_IDLE:
                try:
                    self.node.wait_until(
                        lambda: self.node.deploy_status.controller_state
                        == DeployStatus.CONTROLLER_STATE_IDLE,
                        timeout=min(self.args.transition_timeout, deadline - time.monotonic()),
                        description="IDLE after PRE_IDLE",
                        fail_on_estop=True,
                    )
                except SafetyStop:
                    pass
                continue

            if state == DeployStatus.CONTROLLER_STATE_IDLE:
                self.node.get_logger().info("L2+A: IDLE -> FIX_STAND")
                self.pulse(KEY_FIX_STAND)
                continue

            name = self.CONTROLLER_NAMES.get(state, f"UNKNOWN({state})")
            raise EvaluationError(f"unsupported deploy controller state: {name}")

        raise TransitionTimeout("could not establish ready FIX_STAND")

    def reset_and_wait_ready(
        self, reset_offsets: tuple[float, float, float]
    ) -> int:
        deploy = self.node.deploy_status
        if (
            not deploy.safety_ok
            or deploy.controller_state != DeployStatus.CONTROLLER_STATE_FIX_STAND
            or not deploy.fix_stand_ready
        ):
            raise EvaluationError("reset requested without a ready FIX_STAND")

        previous_episode_id = self.node.episode_status.episode_id
        response = self.node.call_reset(
            self.args.service_timeout,
            base_x_offset=reset_offsets[0],
            base_y_offset=reset_offsets[1],
            base_yaw_offset=reset_offsets[2],
        )
        reset_stamp_ns = stamp_to_ns(response.reset_stamp)
        reset_received_at = time.monotonic()
        if response.episode_id <= previous_episode_id:
            raise EvaluationError(
                "ResetEpisode did not advance episode_id "
                f"({previous_episode_id} -> {response.episode_id})"
            )

        self.node.get_logger().info(
            f"reset episode {response.episode_id}: "
            f"dx={reset_offsets[0]:+.4f} m, dy={reset_offsets[1]:+.4f} m, "
            f"dyaw={reset_offsets[2]:+.4f} rad"
        )

        def reset_is_ready() -> bool:
            episode = self.node.episode_status
            deploy_status = self.node.deploy_status
            reset_age = time.monotonic() - reset_received_at
            if reset_age >= self.args.status_timeout:
                if not episode.depth_ready:
                    raise EvaluationError(
                        "MuJoCo did not publish a post-reset depth frame"
                    )
                if self.node.depth_stamp_ns < reset_stamp_ns:
                    raise EvaluationError(
                        "processed depth stamp did not advance past the simulator reset"
                    )
            return (
                episode.episode_id == response.episode_id
                and episode.outcome == EpisodeStatus.WAITING
                and episode.depth_ready
                and stamp_to_ns(episode.depth_stamp) >= reset_stamp_ns
                and stamp_to_ns(episode.stamp) > reset_stamp_ns
                and abs(episode.roll) <= self.args.max_settle_tilt
                and abs(episode.pitch) <= self.args.max_settle_tilt
                and episode.angular_velocity_rad_s <= self.args.max_settle_angular_velocity
                and episode.linear_velocity_m_s <= self.args.max_settle_linear_velocity
                and deploy_status.safety_ok
                and deploy_status.controller_state
                == DeployStatus.CONTROLLER_STATE_FIX_STAND
                and deploy_status.fix_stand_ready
                and self.node.depth_stamp_ns >= reset_stamp_ns
                and reset_age >= self.args.settle_time
            )

        try:
            self.node.wait_until(
                reset_is_ready,
                timeout=self.args.reset_ready_timeout,
                description="reset state, fresh depth, and settled robot",
                fail_on_estop=True,
            )
        except SafetyStop as error:
            raise TrialFailure("DEPLOY_ESTOP", str(error)) from error
        except TransitionTimeout as error:
            raise TrialFailure("RESET_READY_TIMEOUT", str(error)) from error
        return response.episode_id

    def enter_policy(self) -> None:
        baseline_reset_sequence = self.node.deploy_status.policy_reset_sequence
        self.node.get_logger().info("START: FIX_STAND -> HIGH_CONTROLLER")
        self.pulse(KEY_START)

        def policy_is_ready() -> bool:
            status = self.node.deploy_status
            return (
                status.controller_state == DeployStatus.CONTROLLER_STATE_HIGH_CONTROLLER
                and status.policy_reset_sequence > baseline_reset_sequence
                and status.policy_output_valid
            )

        try:
            self.node.wait_until(
                policy_is_ready,
                timeout=self.args.transition_timeout,
                description="reset and valid HIGH_CONTROLLER policy",
                fail_on_estop=True,
            )
        except SafetyStop as error:
            raise TrialFailure("DEPLOY_ESTOP", str(error)) from error
        except TransitionTimeout as error:
            raise TrialFailure("POLICY_START_TIMEOUT", str(error)) from error

    def arm_episode(self, episode_id: int) -> None:
        response = self.node.call_start(self.args.service_timeout)
        if response.episode_id != episode_id:
            raise EvaluationError(
                f"StartEpisode returned episode {response.episode_id}, expected {episode_id}"
            )
        try:
            self.node.wait_until(
                lambda: self.node.episode_status.episode_id == episode_id
                and self.node.episode_status.outcome == EpisodeStatus.RUNNING,
                timeout=self.args.service_timeout,
                description=f"episode {episode_id} RUNNING",
                fail_on_estop=True,
            )
        except SafetyStop as error:
            raise TrialFailure("DEPLOY_ESTOP", str(error)) from error
        except TransitionTimeout as error:
            raise TrialFailure("EPISODE_START_TIMEOUT", str(error)) from error

    def run_episode(self, episode_id: int) -> EpisodeStatus:
        self.node.set_command(ly=self.args.velocity)
        deadline = time.monotonic() + self.args.episode_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            self.node.spin_once(min(0.02, deadline - time.monotonic()))
            self.node.assert_fresh()
            deploy = self.node.deploy_status
            episode = self.node.episode_status

            if episode.episode_id != episode_id:
                raise EvaluationError(
                    f"episode_id changed while running ({episode_id} -> {episode.episode_id})"
                )
            if episode.outcome in self.TERMINAL_OUTCOMES:
                return episode
            if episode.outcome != EpisodeStatus.RUNNING:
                name = self.OUTCOME_NAMES.get(episode.outcome, str(episode.outcome))
                raise EvaluationError(f"episode left RUNNING without a terminal outcome: {name}")
            if not deploy.safety_ok:
                raise TrialFailure("DEPLOY_ESTOP", deploy.last_fault or "safety flag is false")
            if (
                deploy.controller_state
                != DeployStatus.CONTROLLER_STATE_HIGH_CONTROLLER
            ):
                state = self.CONTROLLER_NAMES.get(
                    deploy.controller_state, str(deploy.controller_state)
                )
                raise TrialFailure(
                    "CONTROLLER_LEFT_HIGH", f"controller state changed to {state}"
                )
            if not deploy.policy_output_valid:
                raise TrialFailure(
                    "POLICY_OUTPUT_INVALID", deploy.last_fault or "policy output became invalid"
                )

        raise TrialFailure(
            "EVALUATOR_TIMEOUT",
            f"no terminal EpisodeStatus within {self.args.episode_timeout:.3f} s",
        )

    def episode_result(
        self,
        *,
        run_id: str,
        trial: int,
        episode_id: int,
        started_at: str,
        wall_duration: float,
        reset_offsets: tuple[float, float, float],
        result: str,
        detail: str,
    ) -> dict[str, object]:
        episode = self.node.episode_status
        simulator_outcome = self.OUTCOME_NAMES.get(
            episode.outcome, f"UNKNOWN({episode.outcome})"
        )
        return {
            "schema_version": 1,
            "run_id": run_id,
            "trial": trial,
            "episode_id": episode_id,
            "started_at": started_at,
            "finished_at": utc_now(),
            "wall_duration_s": wall_duration,
            "command_ly": self.args.velocity,
            "seed": self.args.seed,
            "trial_seed": self.args.seed + trial,
            "reset_base_x_offset_m": reset_offsets[0],
            "reset_base_y_offset_m": reset_offsets[1],
            "reset_base_yaw_offset_rad": reset_offsets[2],
            "result": result,
            "success": result == "SUCCESS",
            "detail": detail,
            "simulator_outcome": simulator_outcome,
            "simulator_outcome_code": int(episode.outcome),
            "simulation_elapsed_s": episode.elapsed_time_s,
            "base_roll_pitch_rad": [episode.roll, episode.pitch],
            "angular_velocity_rad_s": episode.angular_velocity_rad_s,
            "linear_velocity_m_s": episode.linear_velocity_m_s,
            "simulator_message": episode.message,
            "deploy_last_fault": self.node.deploy_status.last_fault,
        }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously evaluate the beam policy in a resident Unitree MuJoCo process."
    )
    parser.add_argument("--trials", type=positive_int, required=True)
    parser.add_argument("--velocity", type=velocity_value, required=True)
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument(
        "--reset-position-jitter-m", type=nonnegative_float, default=0.0
    )
    parser.add_argument(
        "--reset-yaw-jitter-rad", type=nonnegative_float, default=0.0
    )
    parser.add_argument("--heartbeat-hz", type=positive_float, default=50.0)
    parser.add_argument("--button-hold", type=positive_float, default=0.10)
    parser.add_argument("--startup-timeout", type=positive_float, default=15.0)
    parser.add_argument("--status-timeout", type=positive_float, default=1.0)
    parser.add_argument("--service-timeout", type=positive_float, default=5.0)
    parser.add_argument("--transition-timeout", type=positive_float, default=6.0)
    parser.add_argument("--reset-ready-timeout", type=positive_float, default=10.0)
    parser.add_argument("--settle-time", type=positive_float, default=1.0)
    parser.add_argument("--max-settle-tilt", type=positive_float, default=0.15)
    parser.add_argument(
        "--max-settle-angular-velocity", type=positive_float, default=0.30
    )
    parser.add_argument(
        "--max-settle-linear-velocity", type=positive_float, default=0.05
    )
    parser.add_argument("--episode-timeout", type=positive_float, default=25.0)
    return parser.parse_args(argv)


def result_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    stem = (
        f"eval-sim2sim-vx{args.velocity:g}"
        f"-reset-pos{args.reset_position_jitter_m:g}m"
        f"-reset-yaw{args.reset_yaw_jitter_rad:g}rad"
        f"-seed{args.seed}-n{args.trials}"
    )
    return Path(f"{stem}.jsonl"), Path(f"{stem}.summary.json")


def write_summary(
    path: Path,
    *,
    jsonl_path: Path,
    run_id: str,
    args: argparse.Namespace,
    domain_id: int,
    started_at: str,
    records: list[dict[str, object]],
    fatal_error: str | None,
) -> dict[str, object]:
    successes = sum(bool(record["success"]) for record in records)
    counts = Counter(str(record["result"]) for record in records)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": utc_now(),
        "ros_domain_id": domain_id,
        "requested_trials": args.trials,
        "completed_trials": len(records),
        "successes": successes,
        "success_rate": successes / len(records) if records else None,
        "result_counts": dict(sorted(counts.items())),
        "command_ly": args.velocity,
        "seed": args.seed,
        "reset_position_jitter_m": args.reset_position_jitter_m,
        "reset_yaw_jitter_rad": args.reset_yaw_jitter_rad,
        "settle_time_s": args.settle_time,
        "max_settle_tilt_rad": args.max_settle_tilt,
        "max_settle_angular_velocity_rad_s": args.max_settle_angular_velocity,
        "max_settle_linear_velocity_m_s": args.max_settle_linear_velocity,
        "episode_timeout_s": args.episode_timeout,
        "fatal_error": fatal_error,
        "jsonl": str(jsonl_path.resolve()),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        domain_id = validate_simulation_environment()
    except EvaluationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    output_path, summary_path = result_paths(args)
    if output_path.exists():
        print(f"error: generated output already exists: {output_path}", file=sys.stderr)
        return 2
    if summary_path.exists():
        print(f"error: generated summary already exists: {summary_path}", file=sys.stderr)
        return 2
    print(f"[INFO] Trial results: {output_path.resolve()}")
    print(f"[INFO] Summary: {summary_path.resolve()}")

    run_id = str(uuid.uuid4())
    run_started_at = utc_now()
    records: list[dict[str, object]] = []
    fatal_error: str | None = None
    exit_code = 0
    node: EvaluationNode | None = None

    with output_path.open("x", encoding="utf-8", buffering=1) as output_file:
        rclpy.init(args=[])
        try:
            node = EvaluationNode(args.heartbeat_hz, args.status_timeout)
            evaluator = BeamEvaluator(node, args)
            node.wait_for_preflight(args.startup_timeout)
            node.publishing_enabled = True
            node.set_command()
            node.pump(args.button_hold)
            evaluator.normalize_to_fixed_stand()

            for trial in range(args.trials):
                trial_started_at = utc_now()
                trial_started_mono = time.monotonic()
                rng = np.random.default_rng(args.seed + trial)
                reset_offsets = (
                    float(
                        rng.uniform(
                            -args.reset_position_jitter_m,
                            args.reset_position_jitter_m,
                        )
                    ),
                    float(
                        rng.uniform(
                            -args.reset_position_jitter_m,
                            args.reset_position_jitter_m,
                        )
                    ),
                    float(
                        rng.uniform(
                            -args.reset_yaw_jitter_rad,
                            args.reset_yaw_jitter_rad,
                        )
                    ),
                )
                episode_id = node.episode_status.episode_id
                result_name: str
                detail = ""
                try:
                    episode_id = evaluator.reset_and_wait_ready(reset_offsets)
                    evaluator.enter_policy()
                    evaluator.arm_episode(episode_id)
                    terminal = evaluator.run_episode(episode_id)
                    result_name = evaluator.OUTCOME_NAMES[terminal.outcome]
                    detail = terminal.message
                except TrialFailure as error:
                    episode_id = node.episode_status.episode_id
                    result_name = error.result
                    detail = error.detail
                    node.get_logger().error(
                        f"trial {trial + 1}/{args.trials} failed: {result_name}: {detail}"
                    )
                finally:
                    node.set_command()
                    node.pump(args.button_hold)

                record = evaluator.episode_result(
                    run_id=run_id,
                    trial=trial,
                    episode_id=episode_id,
                    started_at=trial_started_at,
                    wall_duration=time.monotonic() - trial_started_mono,
                    reset_offsets=reset_offsets,
                    result=result_name,
                    detail=detail,
                )
                records.append(record)
                output_file.write(json.dumps(record, sort_keys=True) + "\n")
                node.get_logger().info(
                    f"trial {trial + 1}/{args.trials}: {result_name} "
                    f"(episode {episode_id}, {record['wall_duration_s']:.3f} s wall)"
                )
                evaluator.normalize_to_fixed_stand()

        except KeyboardInterrupt:
            fatal_error = "interrupted"
            exit_code = 130
        except (EvaluationError, ExternalShutdownException) as error:
            fatal_error = str(error)
            print(f"error: {error}", file=sys.stderr)
            exit_code = 1
        finally:
            if node is not None:
                if rclpy.ok() and node.publishing_enabled:
                    node.set_command()
                    node.pump(max(args.button_hold, 0.1))
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    summary = write_summary(
        summary_path,
        jsonl_path=output_path,
        run_id=run_id,
        args=args,
        domain_id=domain_id,
        started_at=run_started_at,
        records=records,
        fatal_error=fatal_error,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
