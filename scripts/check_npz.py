#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def _fmt_shape(shape) -> str:
    return "(" + ", ".join(str(int(x)) for x in shape) + ")"


def _main() -> int:
    np.set_printoptions(precision=2, suppress=True, floatmode="fixed")

    parser = argparse.ArgumentParser(
        description=(
            "Quickly inspect a .npz motion file: print keys, shapes, dtypes, and basic consistency checks."
        )
    )
    parser.add_argument("npz", type=Path, help="Path to .npz file")
    args = parser.parse_args()

    path: Path = args.npz
    data = np.load(path)
    print(f"file: {path}")
    keys = list(data.files)
    
    print(f"num_arrays: {len(keys)}")
    for k in keys:
        a = data[k]
        print(f"  - {k}: shape={_fmt_shape(a.shape)} dtype={a.dtype}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
