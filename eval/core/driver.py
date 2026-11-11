"""Controller state-machine driver shared by every task evaluator.

It turns the deployment node's FSM into a stable starting point:
IDLE -> FIX_STAND -> HIGH_CONTROLLER, plus episode arming.  Task evaluators
subclass this and add their own trial loop.
"""

from __future__ import annotations

import time
from typing import Any

import rclpy
from legged_rl_deploy.msg import DeployStatus
from unitree_mujoco.msg import EpisodeStatus

from core.errors import EvaluationError, SafetyStop, TransitionTimeout, TrialFailure
from core.keys import KEY_CLEAR_ESTOP, KEY_FIX_STAND, KEY_SELECT, KEY_START
from core.node import EvaluationNode


class ControllerDriver:
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

    def __init__(self, node: EvaluationNode, args: Any) -> None:
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
