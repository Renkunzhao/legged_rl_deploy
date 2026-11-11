"""Task-agnostic attitude metrics.

Task-specific scoring (stand pose, flight angle, chain composition) lives with
the task evaluator, not here.
"""

from __future__ import annotations

import math

import numpy as np


def rotation(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def body_tilt(q) -> float:
    return math.acos(float(np.clip(rotation(q)[2, 2], -1.0, 1.0)))


def yaw_twist(q) -> float | None:
    """Heading from the world-z swing-twist, robust while the body is pitched.

    Euler yaw (atan2(R[1,0], R[0,0])) flips by pi once a flip passes 90 degrees.
    The twist about world z is the yaw component of the same orientation and is
    stable for a pure pitch/roll swing.  At an exact 180-degree pitch the twist
    is undefined (w and z both zero); return None so the caller can fail the
    measurement instead of inventing a heading.
    """
    w, x, y, z = (float(v) for v in q)
    norm = math.hypot(w, z)
    if norm < 1.0e-6:
        return None
    return wrap_to_pi(2.0 * math.atan2(z, w))
