#!/usr/bin/env python3
"""Repeated beam (bridge) evaluations against a resident Unitree MuJoCo stack.

The simulator's C++ beam episode monitor decides each trial's outcome; this
script only drives the episode loop and aggregates results.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import rclpy
from legged_rl_deploy.msg import DeployStatus
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from unitree_mujoco.msg import EpisodeStatus

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "eval"))

from core.env import validate_simulation_environment
from core.errors import EvaluationError
from core.driver import ControllerDriver
from core.node import EvaluationNode
from core.util import (
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
    stamp_to_ns,
    utc_now,
)

PROCESSED_DEPTH_TOPIC = "/unitree_go2_beam_depth/depth_m"


def velocity_value(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0.5 <= value <= 0.8:
        raise argparse.ArgumentTypeError("must be in [0.5, 0.8]")
    return value


class BeamEvaluationNode(EvaluationNode):
    """Adds the processed-depth topic that the beam policy consumes."""

    def __init__(self, heartbeat_hz: float, status_timeout: float,
                 node_name: str = "beam_sim2sim_evaluator") -> None:
        super().__init__(heartbeat_hz, status_timeout, node_name)
        self.depth_stamp_ns: int | None = None
        self.depth_received_at: float | None = None
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.depth_subscription = self.create_subscription(
            Image, PROCESSED_DEPTH_TOPIC, self._on_depth, sensor_qos
        )

    def _on_depth(self, message: Image) -> None:
        self.depth_stamp_ns = stamp_to_ns(message.header.stamp)
        self.depth_received_at = time.monotonic()

    def extra_freshness(self) -> dict[str, float | None]:
        return {"processed depth": self.depth_received_at}


class BeamEvaluator(ControllerDriver):
    """Bridge task: the simulator monitor owns the terminal outcome."""


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
            node = BeamEvaluationNode(args.heartbeat_hz, args.status_timeout)
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
