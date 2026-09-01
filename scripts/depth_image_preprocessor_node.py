#!/usr/bin/python3
"""Apply a YAML-configured preprocessing pipeline to a ROS depth image."""

from __future__ import annotations

import array
import math
import time
from typing import Sequence, Tuple

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image

_PROCESS_ORDER = (
    "replace_invalid",
    "clip",
    "center_crop",
    "resize_nearest",
    "affine",
)

_REQUIRED_PARAMETERS = {
    "input.topic": Parameter.Type.STRING,
    "input.width": Parameter.Type.INTEGER,
    "input.height": Parameter.Type.INTEGER,
    "input.depth_scale_16uc1": Parameter.Type.DOUBLE,
    "input.camera_info.topic": Parameter.Type.STRING,
    "input.camera_info.intrinsics": Parameter.Type.DOUBLE_ARRAY,
    "input.camera_info.intrinsics_atol": Parameter.Type.DOUBLE,
    "input.camera_info.distortion_model": Parameter.Type.STRING,
    "input.camera_info.distortion": Parameter.Type.DOUBLE_ARRAY,
    "input.camera_info.distortion_atol": Parameter.Type.DOUBLE,
    "output.topic": Parameter.Type.STRING,
    "process_order": Parameter.Type.STRING_ARRAY,
}

_OPERATION_PARAMETERS = {
    "replace_invalid": {
        "replace_invalid.valid_min": Parameter.Type.DOUBLE,
        "replace_invalid.valid_min_inclusive": Parameter.Type.BOOL,
        "replace_invalid.value": Parameter.Type.DOUBLE,
    },
    "clip": {
        "clip.min": Parameter.Type.DOUBLE,
        "clip.max": Parameter.Type.DOUBLE,
    },
    "center_crop": {
        "center_crop.width": Parameter.Type.INTEGER,
        "center_crop.height": Parameter.Type.INTEGER,
    },
    "resize_nearest": {
        "resize_nearest.width": Parameter.Type.INTEGER,
        "resize_nearest.height": Parameter.Type.INTEGER,
    },
    "affine": {
        "affine.scale": Parameter.Type.DOUBLE,
        "affine.offset": Parameter.Type.DOUBLE,
    },
}

_ROS_PARAMETERS = {"use_sim_time", "start_type_description_service"}

_SENSOR_DATA_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def decode_depth_image(message: Image, depth_scale_16uc1: float) -> np.ndarray:
    """Decode a tightly or row-padded 16UC1/32FC1 image as float32 meters."""
    if message.encoding == "16UC1":
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
        scale = depth_scale_16uc1
    elif message.encoding == "32FC1":
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        scale = 1.0
    else:
        raise ValueError(f"unsupported depth encoding: {message.encoding}")

    if message.width == 0 or message.height == 0:
        raise ValueError("depth image dimensions must be nonzero")
    row_bytes = message.width * dtype.itemsize
    required_bytes = message.step * message.height
    if message.step < row_bytes or len(message.data) != required_bytes:
        raise ValueError("malformed depth image buffer")

    values = np.ndarray(
        shape=(message.height, message.width),
        dtype=dtype,
        buffer=memoryview(message.data),
        strides=(message.step, dtype.itemsize),
    )
    return values.astype(np.float32) * scale


def validate_camera_info(
    message: CameraInfo,
    *,
    width: int,
    height: int,
    intrinsics: np.ndarray,
    intrinsics_atol: float,
    distortion_model: str,
    distortion: np.ndarray,
    distortion_atol: float,
) -> None:
    """Validate a CameraInfo message against the configured camera contract."""
    if message.width != width or message.height != height:
        raise ValueError(
            f"camera_info resolution must be {width}x{height}, "
            f"got {message.width}x{message.height}"
        )
    if message.distortion_model != distortion_model:
        raise ValueError(
            f"camera_info distortion_model must be {distortion_model}, "
            f"got {message.distortion_model}"
        )
    if len(message.k) != 9:
        raise ValueError("camera_info K must contain 9 values")

    actual_intrinsics = np.array(
        [message.k[0], message.k[4], message.k[2], message.k[5]],
        dtype=np.float64,
    )
    if not np.allclose(
        actual_intrinsics,
        intrinsics,
        rtol=0.0,
        atol=intrinsics_atol,
    ):
        raise ValueError(
            "camera_info intrinsics do not match: "
            f"expected {intrinsics.tolist()}, got {actual_intrinsics.tolist()}"
        )

    actual_distortion = np.asarray(message.d, dtype=np.float64)
    if actual_distortion.shape != distortion.shape or not np.allclose(
        actual_distortion,
        distortion,
        rtol=0.0,
        atol=distortion_atol,
    ):
        raise ValueError(
            "camera_info distortion does not match: "
            f"expected {distortion.tolist()}, got {actual_distortion.tolist()}"
        )


def center_crop(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return a centered crop with the requested dimensions."""
    source_height, source_width = depth.shape
    if width > source_width or height > source_height:
        raise ValueError(
            f"center_crop {width}x{height} exceeds input {source_width}x{source_height}"
        )
    x = (source_width - width) // 2
    y = (source_height - height) // 2
    return depth[y : y + height, x : x + width]


def resize_nearest(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize with the pixel-center nearest-neighbor rule used during deployment."""
    source_height, source_width = depth.shape
    source_x = np.minimum(
        ((np.arange(width) + 0.5) * source_width / width).astype(np.int64),
        source_width - 1,
    )
    source_y = np.minimum(
        ((np.arange(height) + 0.5) * source_height / height).astype(np.int64),
        source_height - 1,
    )
    return depth[source_y[:, None], source_x[None, :]]


class DepthImagePreprocessorNode(Node):
    def __init__(self) -> None:
        super().__init__(
            "depth_image_preprocessor_node",
            automatically_declare_parameters_from_overrides=True,
            start_parameter_services=False,
        )
        self._validate_parameter_names()

        self.input_topic = self._required("input.topic")
        self.input_width = self._required("input.width")
        self.input_height = self._required("input.height")
        self.depth_scale_16uc1 = self._required("input.depth_scale_16uc1")
        self.camera_info_topic = self._required("input.camera_info.topic")
        self.camera_intrinsics = np.asarray(
            self._required("input.camera_info.intrinsics"), dtype=np.float64
        )
        self.camera_intrinsics_atol = self._required(
            "input.camera_info.intrinsics_atol"
        )
        self.camera_distortion_model = self._required(
            "input.camera_info.distortion_model"
        )
        self.camera_distortion = np.asarray(
            self._required("input.camera_info.distortion"), dtype=np.float64
        )
        self.camera_distortion_atol = self._required(
            "input.camera_info.distortion_atol"
        )
        self.output_topic = self._required("output.topic")
        self.process_order = tuple(self._required("process_order"))

        self._validate_common_parameters()
        self._validate_process_order()
        self._load_operation_parameters()
        self.output_width = self.input_width
        self.output_height = self.input_height
        if "center_crop" in self.process_order:
            self.output_width = self.center_crop_width
            self.output_height = self.center_crop_height
        if "resize_nearest" in self.process_order:
            self.output_width = self.resize_nearest_width
            self.output_height = self.resize_nearest_height

        self._u16_pipeline_is_noop = self.process_order == (
            "replace_invalid",
        ) and (
            (
                self.replace_invalid_valid_min_inclusive
                and self.replace_invalid_valid_min <= 0.0
            )
            or (
                not self.replace_invalid_valid_min_inclusive
                and self.replace_invalid_valid_min < 0.0
            )
        )

        self._output_message = Image()
        self._output_message.height = self.output_height
        self._output_message.width = self.output_width
        self._output_message.encoding = "32FC1"
        self._output_message.is_bigendian = False
        self._output_message.step = self.output_width * np.dtype(np.float32).itemsize
        output_size = self.output_height * self._output_message.step
        self._output_message.data = array.array("B", [0]) * output_size
        self._output_buffer_view = memoryview(self._output_message.data)

        self._stats_arrival_ns = []
        self._stats_input_age_ms = []
        self._stats_process_ms = []
        self._stats_timer = self.create_timer(1.0, self._log_statistics)

        self.camera_info_valid = False
        self.depth_publisher = self.create_publisher(
            Image, self.output_topic, _SENSOR_DATA_QOS
        )
        self._camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._on_camera_info,
            _SENSOR_DATA_QOS,
        )
        self.create_subscription(
            Image,
            self.input_topic,
            self._on_depth,
            _SENSOR_DATA_QOS,
        )
        self.get_logger().info(
            f"configured {self.input_width}x{self.input_height} -> "
            f"{self.output_width}x{self.output_height} depth pipeline "
            f"{list(self.process_order)} for {self.input_topic} -> "
            f"{self.output_topic}"
        )

    @staticmethod
    def _p95_and_max(values: Sequence[float]) -> Tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        return float(np.percentile(values, 95)), max(values)

    def _log_statistics(self) -> None:
        arrival_ns = self._stats_arrival_ns
        input_age_ms = self._stats_input_age_ms
        process_ms = self._stats_process_ms
        self._stats_arrival_ns = []
        self._stats_input_age_ms = []
        self._stats_process_ms = []

        hz = 0.0
        if len(arrival_ns) >= 2:
            span_sec = (arrival_ns[-1] - arrival_ns[0]) * 1.0e-9
            if span_sec > 0.0:
                hz = (len(arrival_ns) - 1) / span_sec
        input_p95, input_max = self._p95_and_max(input_age_ms)
        process_p95, process_max = self._p95_and_max(process_ms)
        self.get_logger().info(
            f"[depth_preprocessor] hz={hz:.1f} "
            f"input_age_ms(p95/max)={input_p95:.2f}/{input_max:.2f} "
            f"process_ms(p95/max)={process_p95:.2f}/{process_max:.2f}"
        )

    def _validate_parameter_names(self) -> None:
        allowed = set(_REQUIRED_PARAMETERS)
        for parameters in _OPERATION_PARAMETERS.values():
            allowed.update(parameters)
        if hasattr(self, "list_parameters"):
            actual = set(self.list_parameters([], depth=10).names)
        else:
            actual = set(self.get_parameters_by_prefix(""))
        unknown = sorted(actual - allowed - _ROS_PARAMETERS)
        if unknown:
            raise ValueError(f"unknown parameters: {unknown}")

    def _required(self, name: str):
        expected_type = _REQUIRED_PARAMETERS.get(name)
        if expected_type is None:
            for parameters in _OPERATION_PARAMETERS.values():
                if name in parameters:
                    expected_type = parameters[name]
                    break
        if not self.has_parameter(name):
            raise ValueError(f"missing required parameter: {name}")
        parameter = self.get_parameter(name)
        if parameter.type_ != expected_type:
            raise ValueError(
                f"parameter {name} must have type {expected_type.name.lower()}, "
                f"got {parameter.type_.name.lower()}"
            )
        return parameter.value

    def _validate_common_parameters(self) -> None:
        topics = (self.input_topic, self.camera_info_topic, self.output_topic)
        if any(not topic for topic in topics):
            raise ValueError("input, camera_info, and output topics must not be empty")
        if len(set(topics)) != len(topics):
            raise ValueError("input, camera_info, and output topics must differ")
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError("input width and height must be positive")
        if not math.isfinite(self.depth_scale_16uc1) or self.depth_scale_16uc1 <= 0.0:
            raise ValueError("input.depth_scale_16uc1 must be finite and positive")
        if self.camera_intrinsics.shape != (4,):
            raise ValueError("input.camera_info.intrinsics must be [fx, fy, cx, cy]")
        if not np.all(np.isfinite(self.camera_intrinsics)):
            raise ValueError("input.camera_info.intrinsics must be finite")
        fx, fy, cx, cy = self.camera_intrinsics
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("camera fx and fy must be positive")
        if not (0.0 <= cx < self.input_width and 0.0 <= cy < self.input_height):
            raise ValueError("camera cx and cy must lie inside the input image")
        if (
            not math.isfinite(self.camera_intrinsics_atol)
            or self.camera_intrinsics_atol < 0.0
        ):
            raise ValueError("camera intrinsics_atol must be finite and nonnegative")
        if not self.camera_distortion_model:
            raise ValueError("camera distortion_model must not be empty")
        if self.camera_distortion.ndim != 1 or self.camera_distortion.size == 0:
            raise ValueError("camera distortion must be a nonempty array")
        if not np.all(np.isfinite(self.camera_distortion)):
            raise ValueError("camera distortion must be finite")
        if (
            not math.isfinite(self.camera_distortion_atol)
            or self.camera_distortion_atol < 0.0
        ):
            raise ValueError("camera distortion_atol must be finite and nonnegative")

    def _validate_process_order(self) -> None:
        if not self.process_order:
            raise ValueError("process_order must not be empty")
        if len(set(self.process_order)) != len(self.process_order):
            raise ValueError("process_order must not contain duplicate operations")
        unknown = [
            operation
            for operation in self.process_order
            if operation not in _PROCESS_ORDER
        ]
        if unknown:
            raise ValueError(f"unknown process operations: {unknown}")
        positions = [
            _PROCESS_ORDER.index(operation) for operation in self.process_order
        ]
        if positions != sorted(positions):
            raise ValueError(
                "process_order must follow: " + " -> ".join(_PROCESS_ORDER)
            )

    def _load_operation_parameters(self) -> None:
        for operation, parameters in _OPERATION_PARAMETERS.items():
            enabled = operation in self.process_order
            present = [name for name in parameters if self.has_parameter(name)]
            if enabled and len(present) != len(parameters):
                missing = sorted(set(parameters) - set(present))
                raise ValueError(f"{operation} is missing parameters: {missing}")
            if not enabled and present:
                raise ValueError(
                    f"parameters for disabled operation {operation}: {sorted(present)}"
                )
            for name in present:
                setattr(self, name.replace(".", "_"), self._required(name))

        if "replace_invalid" in self.process_order:
            if not math.isfinite(self.replace_invalid_valid_min):
                raise ValueError("replace_invalid.valid_min must be finite")
            if not math.isfinite(self.replace_invalid_value):
                raise ValueError("replace_invalid.value must be finite")

        if "clip" in self.process_order and (
            not math.isfinite(self.clip_min)
            or not math.isfinite(self.clip_max)
            or self.clip_min >= self.clip_max
        ):
            raise ValueError("clip requires finite min < max")

        if "center_crop" in self.process_order:
            if self.center_crop_width <= 0 or self.center_crop_height <= 0:
                raise ValueError("center_crop width and height must be positive")
            if (
                self.center_crop_width > self.input_width
                or self.center_crop_height > self.input_height
            ):
                raise ValueError("center_crop dimensions exceed the input image")

        if "resize_nearest" in self.process_order and (
            self.resize_nearest_width <= 0 or self.resize_nearest_height <= 0
        ):
            raise ValueError("resize_nearest width and height must be positive")

        if "affine" in self.process_order and (
            not math.isfinite(self.affine_scale)
            or not math.isfinite(self.affine_offset)
        ):
            raise ValueError("affine scale and offset must be finite")

    def _on_camera_info(self, message: CameraInfo) -> None:
        validate_camera_info(
            message,
            width=self.input_width,
            height=self.input_height,
            intrinsics=self.camera_intrinsics,
            intrinsics_atol=self.camera_intrinsics_atol,
            distortion_model=self.camera_distortion_model,
            distortion=self.camera_distortion,
            distortion_atol=self.camera_distortion_atol,
        )
        if not self.camera_info_valid:
            self.get_logger().info("camera_info matches the configured input contract")
        self.camera_info_valid = True
        subscription = self._camera_info_subscription
        if subscription is not None:
            self._camera_info_subscription = None
            self.destroy_subscription(subscription)

    def _on_depth(self, message: Image) -> None:
        callback_start_ns = time.monotonic_ns()
        self._stats_arrival_ns.append(callback_start_ns)
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        if stamp_ns > 0:
            self._stats_input_age_ms.append(
                (self.get_clock().now().nanoseconds - stamp_ns) * 1.0e-6
            )

        if not self.camera_info_valid:
            self.get_logger().warning(
                "waiting for matching camera_info",
                throttle_duration_sec=2.0,
            )
            return
        if message.width != self.input_width or message.height != self.input_height:
            raise ValueError(
                f"depth image resolution must be {self.input_width}x{self.input_height}, "
                f"got {message.width}x{message.height}"
            )

        depth_m = decode_depth_image(message, self.depth_scale_16uc1)
        fast_u16_path = message.encoding == "16UC1" and self._u16_pipeline_is_noop
        if not fast_u16_path:
            for operation in self.process_order:
                if operation == "replace_invalid":
                    invalid = ~np.isfinite(depth_m)
                    if self.replace_invalid_valid_min_inclusive:
                        invalid |= depth_m < self.replace_invalid_valid_min
                    else:
                        invalid |= depth_m <= self.replace_invalid_valid_min
                    depth_m[invalid] = self.replace_invalid_value
                elif operation == "clip":
                    np.clip(depth_m, self.clip_min, self.clip_max, out=depth_m)
                elif operation == "center_crop":
                    depth_m = center_crop(
                        depth_m,
                        self.center_crop_width,
                        self.center_crop_height,
                    )
                elif operation == "resize_nearest":
                    depth_m = resize_nearest(
                        depth_m,
                        self.resize_nearest_width,
                        self.resize_nearest_height,
                    )
                elif operation == "affine":
                    depth_m = depth_m * self.affine_scale + self.affine_offset
                else:
                    raise RuntimeError(f"unimplemented process operation: {operation}")

            if not np.all(np.isfinite(depth_m)):
                raise ValueError("preprocessed depth image contains NaN or Inf")
        depth_m = np.ascontiguousarray(depth_m, dtype="<f4")

        output = self._output_message
        output.header = message.header
        self._output_buffer_view[:] = memoryview(depth_m).cast("B")
        self._stats_process_ms.append(
            (time.monotonic_ns() - callback_start_ns) * 1.0e-6
        )
        self.depth_publisher.publish(output)


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = DepthImagePreprocessorNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
