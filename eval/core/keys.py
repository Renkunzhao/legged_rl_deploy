"""Unitree wireless-controller button bit masks."""

from __future__ import annotations

KEY_START = 1 << 2
KEY_SELECT = 1 << 3
KEY_L2 = 1 << 5
KEY_A = 1 << 8

KEY_CLEAR_ESTOP = KEY_SELECT | KEY_START
KEY_FIX_STAND = KEY_L2 | KEY_A
