"""Validate that the evaluator runs on an isolated loopback ROS graph."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from core.errors import EvaluationError


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
