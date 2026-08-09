#!/usr/bin/python3
"""Convert a standard ROS depth stream to the raw-depth policy contract."""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


_OUTPUT_WIDTH = 96
_OUTPUT_HEIGHT = 72


def decode_depth(message: Image, depth_scale_16uc1: float) -> np.ndarray:
    if message.encoding == "16UC1":
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
        scale = depth_scale_16uc1
    elif message.encoding == "32FC1":
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        scale = 1.0
    else:
        raise ValueError(f"unsupported depth encoding: {message.encoding}")

    row_bytes = message.width * dtype.itemsize
    required_bytes = message.step * message.height
    if message.width == 0 or message.height == 0:
        raise ValueError("depth image dimensions must be nonzero")
    if message.step < row_bytes or len(message.data) < required_bytes:
        raise ValueError("malformed depth image buffer")

    depth = np.ndarray(
        shape=(message.height, message.width),
        dtype=dtype,
        buffer=memoryview(message.data),
        strides=(message.step, dtype.itemsize),
    )
    return depth.astype(np.float32) * scale


def center_crop_and_resize(depth: np.ndarray) -> np.ndarray:
    source_height, source_width = depth.shape
    if source_width * _OUTPUT_HEIGHT > source_height * _OUTPUT_WIDTH:
        crop_height = source_height
        crop_width = source_height * _OUTPUT_WIDTH // _OUTPUT_HEIGHT
    else:
        crop_width = source_width
        crop_height = source_width * _OUTPUT_HEIGHT // _OUTPUT_WIDTH

    crop_x = (source_width - crop_width) // 2
    crop_y = (source_height - crop_height) // 2
    source_x = np.minimum(
        ((np.arange(_OUTPUT_WIDTH) + 0.5) * crop_width / _OUTPUT_WIDTH).astype(
            np.int64
        ),
        crop_width - 1,
    )
    source_y = np.minimum(
        ((np.arange(_OUTPUT_HEIGHT) + 0.5) * crop_height / _OUTPUT_HEIGHT).astype(
            np.int64
        ),
        crop_height - 1,
    )
    return depth[crop_y + source_y[:, None], crop_x + source_x[None, :]].copy()


class DepthImagePreprocessorNode(Node):
    def __init__(self) -> None:
        super().__init__("depth_image_preprocessor_node")
        self.input_topic = self.declare_parameter(
            "input_topic", "/camera/depth/image_rect_raw"
        ).value
        self.output_topic = self.declare_parameter(
            "output_topic", "/bridge_mjlab_raw_depth/depth_image"
        ).value
        self.rate_hz = float(self.declare_parameter("rate_hz", 10.0).value)
        self.input_timeout = float(
            self.declare_parameter("input_timeout_sec", 0.3).value
        )
        self.min_depth = float(self.declare_parameter("min_depth", 0.3).value)
        self.max_depth = float(self.declare_parameter("max_depth", 3.0).value)
        self.depth_scale_16uc1 = float(
            self.declare_parameter("depth_scale_16uc1", 0.001).value
        )

        if self.input_topic == self.output_topic:
            raise ValueError("input_topic and output_topic must differ")
        if self.rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        if self.input_timeout <= 0.0:
            raise ValueError("input_timeout_sec must be positive")
        if self.min_depth <= 0.0 or self.max_depth <= self.min_depth:
            raise ValueError("invalid depth range")
        if self.depth_scale_16uc1 <= 0.0:
            raise ValueError("depth_scale_16uc1 must be positive")

        self.latest_message: Image | None = None
        self.latest_receive_time = 0.0
        self.message_lock = threading.Lock()
        self.depth_publisher = self.create_publisher(Image, self.output_topic, 10)
        self.stats_publisher = self.create_publisher(
            Float32MultiArray, "/bridge_mjlab_raw_depth/camera_stats", 10
        )
        self.create_subscription(
            Image, self.input_topic, self._on_depth, qos_profile_sensor_data
        )
        self.create_timer(1.0 / self.rate_hz, self._publish)
        self.get_logger().info(
            f"preprocessing {self.input_topic} -> {self.output_topic} "
            f"at {_OUTPUT_WIDTH}x{_OUTPUT_HEIGHT}, {self.rate_hz:g} Hz, normalized"
        )

    def _on_depth(self, message: Image) -> None:
        with self.message_lock:
            self.latest_message = message
            self.latest_receive_time = time.monotonic()

    def _publish(self) -> None:
        with self.message_lock:
            message = self.latest_message
            receive_time = self.latest_receive_time
        if message is None:
            return
        if time.monotonic() - receive_time > self.input_timeout:
            self.get_logger().warning(
                "input depth image is stale",
                throttle_duration_sec=2.0,
            )
            return

        try:
            depth_m = decode_depth(message, self.depth_scale_16uc1)
            depth_m = np.nan_to_num(
                depth_m,
                nan=self.max_depth,
                posinf=self.max_depth,
                neginf=self.max_depth,
            )
            depth_m = np.where(depth_m > 0.0, depth_m, self.max_depth)
            depth_m = np.clip(depth_m, self.min_depth, self.max_depth)
            depth_m = center_crop_and_resize(depth_m).astype(np.float32)
            camera = (depth_m / self.max_depth).astype(np.float32)
        except ValueError as exception:
            self.get_logger().warning(
                str(exception),
                throttle_duration_sec=2.0,
            )
            return

        output = Image()
        output.header = message.header
        output.height = _OUTPUT_HEIGHT
        output.width = _OUTPUT_WIDTH
        output.encoding = "32FC1"
        output.is_bigendian = False
        output.step = _OUTPUT_WIDTH * np.dtype(np.float32).itemsize
        output.data = camera.tobytes()
        self.depth_publisher.publish(output)

        stats = Float32MultiArray()
        stats.data = [
            float(depth_m.min()),
            float(depth_m.max()),
            float(depth_m.mean()),
            float(np.mean(depth_m < 1.0)),
            float(np.mean(depth_m >= self.max_depth - 1.0e-3)),
        ]
        self.stats_publisher.publish(stats)


def main() -> None:
    rclpy.init()
    node = DepthImagePreprocessorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
