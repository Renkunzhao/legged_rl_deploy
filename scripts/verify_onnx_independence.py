#!/usr/bin/env python3
"""Verify that 'obs' and 'time_step' are independent in an ONNX model.

If the action output only depends on obs (not time_step), and the motion
outputs (joint_pos, joint_vel, body_quat_w, ...) only depend on time_step
(not obs), then we can safely:
  - Fill obs=0 when using the model as a motion loader
  - Fill time_step=0 when using the model as a policy runner

Usage:
  python3 scripts/verify_onnx_independence.py policies/g1/mimic/pose/policy.onnx
"""

from __future__ import annotations
import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Verify obs/time_step independence in ONNX model.")
    parser.add_argument("onnx", help="Path to .onnx file")
    parser.add_argument("--trials", type=int, default=5, help="Number of random trials")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("pip install onnxruntime", file=sys.stderr)
        return 1

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    inputs = {i.name: i for i in sess.get_inputs()}
    outputs = [o.name for o in sess.get_outputs()]

    if "obs" not in inputs or "time_step" not in inputs:
        print("Model does not have both 'obs' and 'time_step' inputs. Nothing to verify.")
        return 0

    obs_shape = [d if isinstance(d, int) else 1 for d in inputs["obs"].shape]
    ts_shape = [d if isinstance(d, int) else 1 for d in inputs["time_step"].shape]

    action_name = outputs[0]
    motion_names = [n for n in outputs if n != action_name]

    print(f"Action output:  {action_name}")
    print(f"Motion outputs: {motion_names}")
    print(f"obs shape:      {obs_shape}")
    print(f"time_step shape: {ts_shape}")
    print()

    all_ok = True

    for trial in range(args.trials):
        obs_a = np.random.randn(*obs_shape).astype(np.float32)
        obs_b = np.random.randn(*obs_shape).astype(np.float32)
        ts_a = np.random.randn(*ts_shape).astype(np.float32)
        ts_b = np.random.randn(*ts_shape).astype(np.float32)

        # Test 1: action should NOT change when time_step varies (obs fixed)
        r1 = sess.run(outputs, {"obs": obs_a, "time_step": ts_a})
        r2 = sess.run(outputs, {"obs": obs_a, "time_step": ts_b})
        out1 = dict(zip(outputs, r1))
        out2 = dict(zip(outputs, r2))

        action_diff = np.max(np.abs(out1[action_name] - out2[action_name]))
        if action_diff > args.atol:
            print(f"  FAIL trial {trial}: action depends on time_step! max_diff={action_diff:.6e}")
            all_ok = False
        else:
            print(f"  OK   trial {trial}: action independent of time_step (diff={action_diff:.2e})")

        # Test 2: motion outputs should NOT change when obs varies (time_step fixed)
        r3 = sess.run(outputs, {"obs": obs_a, "time_step": ts_a})
        r4 = sess.run(outputs, {"obs": obs_b, "time_step": ts_a})
        out3 = dict(zip(outputs, r3))
        out4 = dict(zip(outputs, r4))

        for mname in motion_names:
            motion_diff = np.max(np.abs(out3[mname] - out4[mname]))
            if motion_diff > args.atol:
                print(f"  FAIL trial {trial}: {mname} depends on obs! max_diff={motion_diff:.6e}")
                all_ok = False
            else:
                print(f"  OK   trial {trial}: {mname} independent of obs (diff={motion_diff:.2e})")

    print()
    if all_ok:
        print("✅ All checks passed: obs and time_step are independent.")
    else:
        print("❌ Some checks failed: obs and time_step may NOT be independent!")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
