"""Shared decoding helpers for ROS depth images."""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image


def decode_depth_values(message: Image) -> np.ndarray:
    """Return a compact copy of a supported depth image."""
    if message.encoding == "16UC1":
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
    elif message.encoding == "32FC1":
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
    else:
        raise ValueError(f"unsupported depth encoding: {message.encoding}")

    if message.width == 0 or message.height == 0:
        raise ValueError("depth image dimensions must be nonzero")
    row_bytes = message.width * dtype.itemsize
    required_bytes = message.step * message.height
    if message.step < row_bytes or len(message.data) < required_bytes:
        raise ValueError("malformed depth image buffer")

    return np.ndarray(
        shape=(message.height, message.width),
        dtype=dtype,
        buffer=memoryview(message.data),
        strides=(message.step, dtype.itemsize),
    ).copy()


def depth_values_to_meters(
    values: np.ndarray,
    encoding: str,
    depth_scale_16uc1: float,
) -> np.ndarray:
    """Convert decoded depth values to float32 meters."""
    scale = depth_scale_16uc1 if encoding == "16UC1" else 1.0
    return values.astype(np.float32) * scale


def decode_depth(message: Image, depth_scale_16uc1: float) -> np.ndarray:
    """Decode a supported ROS depth image as float32 meters."""
    values = decode_depth_values(message)
    return depth_values_to_meters(
        values,
        message.encoding,
        depth_scale_16uc1,
    )
