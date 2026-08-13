#!/usr/bin/python3
"""Add calibrated stereo-occlusion holes to an ideal ROS depth stream."""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from depth_image_utils import decode_depth_values, depth_values_to_meters


def stereo_occlusion_mask(
    depth_m: np.ndarray,
    *,
    focal_length_px: float,
    stereo_baseline_m: float,
    min_depth_jump_m: float,
    max_occlusion_width_px: int,
    occlusion_side: str,
) -> np.ndarray:
    """Return background pixels hidden from the other stereo imager.

    ``occlusion_side`` names the background side of the foreground object in
    image coordinates. The foreground pixels are never included in the mask.
    """
    if depth_m.ndim != 2:
        raise ValueError(f"expected a 2D depth image, got shape {depth_m.shape}")
    if focal_length_px <= 0.0:
        raise ValueError("focal_length_px must be positive")
    if stereo_baseline_m <= 0.0:
        raise ValueError("stereo_baseline_m must be positive")
    if min_depth_jump_m <= 0.0:
        raise ValueError("min_depth_jump_m must be positive")
    if max_occlusion_width_px <= 0:
        raise ValueError("max_occlusion_width_px must be positive")
    if occlusion_side not in ("left", "right"):
        raise ValueError("occlusion_side must be 'left' or 'right'")

    height, image_width = depth_m.shape
    mask = np.zeros((height, image_width), dtype=bool)
    if image_width < 2:
        return mask

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    left = depth_m[:, :-1]
    right = depth_m[:, 1:]
    valid_pair = valid[:, :-1] & valid[:, 1:]

    if occlusion_side == "left":
        far = left
        near = right
    else:
        near = left
        far = right

    edge = valid_pair & ((far - near) >= min_depth_jump_m)
    inverse_near = np.zeros_like(near, dtype=np.float32)
    inverse_far = np.zeros_like(far, dtype=np.float32)
    np.divide(1.0, near, out=inverse_near, where=edge)
    np.divide(1.0, far, out=inverse_far, where=edge)
    inverse_depth_delta = np.zeros_like(far, dtype=np.float32)
    np.subtract(
        inverse_near,
        inverse_far,
        out=inverse_depth_delta,
        where=edge,
    )
    widths = np.ceil(
        focal_length_px * stereo_baseline_m * inverse_depth_delta
    ).astype(np.int32)
    widths = np.where(
        edge,
        np.clip(widths, 1, max_occlusion_width_px),
        0,
    )

    expansion = min(max_occlusion_width_px, image_width - 1)
    for offset in range(expansion):
        active = edge & (widths > offset)
        if occlusion_side == "left":
            mask[:, : image_width - 1 - offset] |= active[:, offset:]
        else:
            mask[:, 1 + offset :] |= active[:, : image_width - 1 - offset]
    return mask


class StereoDepthArtifactNode(Node):
    def __init__(self) -> None:
        super().__init__("stereo_depth_artifact_node")
        self.input_topic = self.declare_parameter(
            "input_topic", "/camera/depth/image_rect_raw"
        ).value
        self.output_topic = self.declare_parameter(
            "output_topic", "/camera/depth/image_rect_raw_stereo_artifact"
        ).value
        self.focal_length_px = float(
            self.declare_parameter("focal_length_px", 389.627960205078).value
        )
        self.stereo_baseline_m = float(
            self.declare_parameter("stereo_baseline_m", 0.050).value
        )
        self.min_depth_jump_m = float(
            self.declare_parameter("min_depth_jump_m", 0.20).value
        )
        self.max_occlusion_width_px = int(
            self.declare_parameter("max_occlusion_width_px", 24).value
        )
        self.occlusion_side = str(
            self.declare_parameter("occlusion_side", "left").value
        )
        self.depth_scale_16uc1 = float(
            self.declare_parameter("depth_scale_16uc1", 0.001).value
        )

        if self.input_topic == self.output_topic:
            raise ValueError("input_topic and output_topic must differ")
        if self.depth_scale_16uc1 <= 0.0:
            raise ValueError("depth_scale_16uc1 must be positive")

        # Validate every artifact parameter before subscribing.
        stereo_occlusion_mask(
            np.ones((1, 2), dtype=np.float32),
            focal_length_px=self.focal_length_px,
            stereo_baseline_m=self.stereo_baseline_m,
            min_depth_jump_m=self.min_depth_jump_m,
            max_occlusion_width_px=self.max_occlusion_width_px,
            occlusion_side=self.occlusion_side,
        )

        self.publisher = self.create_publisher(
            Image,
            self.output_topic,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.input_topic,
            self._on_depth,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"simulating {self.occlusion_side}-side stereo occlusion "
            f"for {self.input_topic} -> {self.output_topic}, "
            f"fx={self.focal_length_px:g} px, "
            f"baseline={self.stereo_baseline_m:g} m"
        )

    def _on_depth(self, message: Image) -> None:
        try:
            values = decode_depth_values(message)
            depth_m = depth_values_to_meters(
                values,
                message.encoding,
                self.depth_scale_16uc1,
            )
            mask = stereo_occlusion_mask(
                depth_m,
                focal_length_px=self.focal_length_px,
                stereo_baseline_m=self.stereo_baseline_m,
                min_depth_jump_m=self.min_depth_jump_m,
                max_occlusion_width_px=self.max_occlusion_width_px,
                occlusion_side=self.occlusion_side,
            )
        except ValueError as exception:
            self.get_logger().warning(
                str(exception),
                throttle_duration_sec=2.0,
            )
            return

        values[mask] = 0
        output = Image()
        output.header = message.header
        output.height = message.height
        output.width = message.width
        output.encoding = message.encoding
        output.is_bigendian = message.is_bigendian
        output.step = message.width * values.dtype.itemsize
        output.data = values.tobytes()
        self.publisher.publish(output)


def main() -> None:
    rclpy.init()
    node = StereoDepthArtifactNode()
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
