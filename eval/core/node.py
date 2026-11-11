"""Shared ROS evaluation node.

The base node knows nothing about a task.  It owns the wireless-controller
publisher, the deploy/episode status subscriptions, the reset/start service
clients, heartbeat, staleness checks, and the "only this evaluator may publish
wireless commands" guard.

A task subclass adds its own subscriptions and reports them through
``extra_freshness()``; the base class then applies the same freshness policy.
"""

from __future__ import annotations

import time
from typing import Callable

import rclpy
from legged_rl_deploy.msg import DeployStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_go.msg import WirelessController
from unitree_mujoco.msg import EpisodeStatus
from unitree_mujoco.srv import ResetEpisode, StartEpisode

from core.errors import EvaluationError, SafetyStop, TransitionTimeout

WIRELESS_TOPIC = "/wirelesscontroller"
DEPLOY_STATUS_TOPIC = "/legged_rl_deploy/status"
EPISODE_STATUS_TOPIC = "/unitree_mujoco/episode_status"
RESET_SERVICE = "/unitree_mujoco/reset_episode"
START_SERVICE = "/unitree_mujoco/start_episode"


class EvaluationNode(Node):
    def __init__(self, heartbeat_hz: float, status_timeout: float,
                 node_name: str = "legged_eval") -> None:
        super().__init__(node_name)
        self.status_timeout = status_timeout
        self.deploy_status: DeployStatus | None = None
        self.episode_status: EpisodeStatus | None = None
        self.deploy_received_at: float | None = None
        self.episode_received_at: float | None = None
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
        self.command_publisher = self.create_publisher(
            WirelessController, WIRELESS_TOPIC, command_qos
        )
        self.deploy_subscription = self.create_subscription(
            DeployStatus, DEPLOY_STATUS_TOPIC, self._on_deploy_status, reliable_qos
        )
        self.episode_subscription = self.create_subscription(
            EpisodeStatus, EPISODE_STATUS_TOPIC, self._on_episode_status, reliable_qos
        )
        self.reset_client = self.create_client(ResetEpisode, RESET_SERVICE)
        self.start_client = self.create_client(StartEpisode, START_SERVICE)

        self.command = WirelessController()
        self.command_timer = self.create_timer(1.0 / heartbeat_hz, self._publish_command)

    # ---- task hooks -----------------------------------------------------

    def extra_freshness(self) -> dict[str, float | None]:
        """Task telemetry that must arrive within ``status_timeout``.

        Keys are human-readable labels used in error messages; values are the
        monotonic receive times (``None`` before the first message).
        """
        return {}

    def preflight_extra_ready(self) -> bool:
        """Task-specific readiness beyond message freshness and services."""
        return True

    # ---- common plumbing -------------------------------------------------

    def _on_deploy_status(self, message: DeployStatus) -> None:
        self.deploy_status = message
        self.deploy_received_at = time.monotonic()

    def _on_episode_status(self, message: EpisodeStatus) -> None:
        self.episode_status = message
        self.episode_received_at = time.monotonic()

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

    def missing_preflight(self) -> list[str]:
        missing = []
        if self.deploy_status is None:
            missing.append(DEPLOY_STATUS_TOPIC)
        if self.episode_status is None:
            missing.append(EPISODE_STATUS_TOPIC)
        if not self.reset_client.service_is_ready():
            missing.append(RESET_SERVICE)
        if not self.start_client.service_is_ready():
            missing.append(START_SERVICE)
        missing.extend(
            label for label, received_at in self.extra_freshness().items()
            if received_at is None
        )
        return missing

    def wait_for_preflight(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            self.spin_once(min(0.05, deadline - time.monotonic()))
            if not self.missing_preflight() and self.preflight_extra_ready():
                self.assert_fresh()
                self.assert_only_wireless_publisher(force=True)
                return
        missing = self.missing_preflight()
        raise EvaluationError(
            "preflight timed out waiting for: "
            + ", ".join(missing if missing else ["task telemetry"])
        )

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
        ):
            raise EvaluationError("deploy or episode status has not been received")
        deploy_age = now - self.deploy_received_at
        episode_age = now - self.episode_received_at
        if deploy_age > self.status_timeout:
            raise EvaluationError(
                f"{DEPLOY_STATUS_TOPIC} is stale ({deploy_age:.3f} s)"
            )
        if episode_age > self.status_timeout:
            raise EvaluationError(
                f"{EPISODE_STATUS_TOPIC} is stale ({episode_age:.3f} s)"
            )
        for label, received_at in self.extra_freshness().items():
            if received_at is None:
                raise EvaluationError(f"{label} has not been received")
            age = now - received_at
            if age > self.status_timeout:
                raise EvaluationError(f"{label} is stale ({age:.3f} s)")
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
