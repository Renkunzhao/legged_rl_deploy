#!/usr/bin/env python3
"""Compare motion from CSV vs ONNX model outputs for the first N frames.

Usage:
    python3 compare_csv_onnx.py <policy.onnx> <motion.csv> [--frames N]

CSV format (per row, 36 cols for G1-29dof):
    pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, j0, j1, ..., j28

ONNX model:
    inputs:  obs [1,154], time_step [1,1]
    outputs: actions [1,29], joint_pos [1,29], joint_vel [1,29],
             body_pos_w [1,14,3], body_quat_w [1,14,4], ...
"""

import argparse
import csv
import sys
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("pip install onnxruntime")
    sys.exit(1)


def load_csv(path):
    """Load motion CSV → list of float arrays (one per frame)."""
    rows = []
    with open(path) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split(",") if x.strip()]
            if vals:
                rows.append(vals)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx", help="Path to policy.onnx")
    parser.add_argument("csv", help="Path to motion.csv")
    parser.add_argument("--frames", type=int, default=10,
                        help="Number of frames to compare (default: 10)")
    args = parser.parse_args()

    # ── load CSV ──
    csv_data = load_csv(args.csv)
    num_csv_frames = len(csv_data)
    n_cols = len(csv_data[0])
    n_joints = n_cols - 7  # 3 pos + 4 quat = 7
    print(f"CSV: {num_csv_frames} frames, {n_cols} cols ({n_joints} joints)")

    # ── load ONNX ──
    sess = ort.InferenceSession(args.onnx)

    in_names = [inp.name for inp in sess.get_inputs()]
    in_shapes = {inp.name: inp.shape for inp in sess.get_inputs()}
    out_names = [out.name for out in sess.get_outputs()]
    print(f"ONNX inputs:  {dict(zip(in_names, [in_shapes[n] for n in in_names]))}")
    print(f"ONNX outputs: {out_names}")
    print()

    # Build input dict: all zeros, we only care about time_step → motion path
    def make_inputs(time_step_val):
        feeds = {}
        for name in in_names:
            shape = in_shapes[name]
            # Replace dynamic dims with 1
            concrete = [s if isinstance(s, int) and s > 0 else 1 for s in shape]
            feeds[name] = np.zeros(concrete, dtype=np.float32)
        if "time_step" in feeds:
            feeds["time_step"][:] = time_step_val
        return feeds

    # ── compare frames ──
    n_compare = min(args.frames, num_csv_frames)

    # Collect per-joint max errors across all frames
    jp_errors = []   # joint_pos errors per frame
    rq_errors = []   # root quat errors per frame
    rp_errors = []   # root pos errors per frame

    for t in range(n_compare):
        feeds = make_inputs(float(t))
        results = dict(zip(out_names, sess.run(out_names, feeds)))

        csv_row = csv_data[t]
        csv_root_pos = np.array(csv_row[0:3], dtype=np.float32)
        # CSV quat: x,y,z,w  →  ONNX body_quat_w: w,x,y,z
        csv_quat_xyzw = np.array(csv_row[3:7], dtype=np.float32)
        csv_quat_wxyz = np.array([csv_quat_xyzw[3], csv_quat_xyzw[0],
                                   csv_quat_xyzw[1], csv_quat_xyzw[2]],
                                  dtype=np.float32)
        csv_joints = np.array(csv_row[7:7+n_joints], dtype=np.float32)

        # ONNX outputs
        onnx_jp = results.get("joint_pos")
        onnx_bq = results.get("body_quat_w")
        onnx_bp = results.get("body_pos_w")

        print(f"── Frame {t} (time_step={float(t):.1f}) ──")

        # Joint positions
        if onnx_jp is not None:
            onnx_jp_flat = onnx_jp.flatten()[:n_joints]
            diff = np.abs(onnx_jp_flat - csv_joints)
            jp_errors.append(diff)
            max_idx = np.argmax(diff)
            print(f"  joint_pos  max|diff|={diff.max():.6f} (joint {max_idx})  "
                  f"mean={diff.mean():.6f}")
            if diff.max() > 0.01:
                print(f"    CSV:  {csv_joints[max_idx]:.6f}")
                print(f"    ONNX: {onnx_jp_flat[max_idx]:.6f}")

        # Root quaternion (body 0)
        if onnx_bq is not None:
            onnx_rq = onnx_bq.reshape(-1, 4)[0]  # body 0, [w,x,y,z]
            # Handle sign ambiguity: q and -q represent same rotation
            if np.dot(onnx_rq, csv_quat_wxyz) < 0:
                onnx_rq = -onnx_rq
            diff_q = np.abs(onnx_rq - csv_quat_wxyz)
            rq_errors.append(diff_q)
            print(f"  root_quat  max|diff|={diff_q.max():.6f}  "
                  f"mean={diff_q.mean():.6f}")
            if diff_q.max() > 0.01:
                print(f"    CSV(wxyz):  {csv_quat_wxyz}")
                print(f"    ONNX(wxyz): {onnx_rq}")

        # Root position (body 0)
        if onnx_bp is not None:
            onnx_rp = onnx_bp.reshape(-1, 3)[0]  # body 0
            diff_p = np.abs(onnx_rp - csv_root_pos)
            rp_errors.append(diff_p)
            print(f"  root_pos   max|diff|={diff_p.max():.6f}  "
                  f"mean={diff_p.mean():.6f}")

        # Print first frame full comparison for reference
        if t == 0:
            print(f"\n  [Frame 0 full joint_pos comparison]")
            if onnx_jp is not None:
                onnx_jp_flat = onnx_jp.flatten()[:n_joints]
                for j in range(n_joints):
                    d = abs(onnx_jp_flat[j] - csv_joints[j])
                    flag = " <<<" if d > 0.01 else ""
                    print(f"    j{j:2d}  csv={csv_joints[j]:+.6f}  "
                          f"onnx={onnx_jp_flat[j]:+.6f}  diff={d:.6f}{flag}")
            print()

    # ── summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if jp_errors:
        all_jp = np.array(jp_errors)
        print(f"  joint_pos:  max_err={all_jp.max():.6f}  "
              f"mean_err={all_jp.mean():.6f}  "
              f"frames_with_err>0.01: {(all_jp.max(axis=1) > 0.01).sum()}/{n_compare}")
    if rq_errors:
        all_rq = np.array(rq_errors)
        print(f"  root_quat:  max_err={all_rq.max():.6f}  "
              f"mean_err={all_rq.mean():.6f}")
    if rp_errors:
        all_rp = np.array(rp_errors)
        print(f"  root_pos:   max_err={all_rp.max():.6f}  "
              f"mean_err={all_rp.mean():.6f}")

    total_max = max(
        (all_jp.max() if jp_errors else 0),
        (all_rq.max() if rq_errors else 0),
    )
    if total_max < 1e-5:
        print("\n✅ CSV and ONNX outputs match perfectly.")
    elif total_max < 0.01:
        print(f"\n⚠️  Small differences (max={total_max:.6f}), likely float precision.")
    else:
        print(f"\n❌ Significant mismatch (max={total_max:.6f})! Check data pipeline.")


if __name__ == "__main__":
    main()
