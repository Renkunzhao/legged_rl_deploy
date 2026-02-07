#!/usr/bin/env python3
"""Print all inputs and outputs of an ONNX model.

Usage:
  python3 scripts/check_onnx.py policy.onnx
"""

import argparse
import sys

ELEM_TYPE_MAP = {
    0: "UNDEFINED", 1: "FLOAT", 2: "UINT8", 3: "INT8", 4: "UINT16",
    5: "INT16", 6: "INT32", 7: "INT64", 8: "STRING", 9: "BOOL",
    10: "FLOAT16", 11: "DOUBLE", 12: "UINT32", 13: "UINT64",
    14: "COMPLEX64", 15: "COMPLEX128", 16: "BFLOAT16",
}


def describe_tensor(info):
    t = info.type.tensor_type
    dtype = ELEM_TYPE_MAP.get(t.elem_type, f"?({t.elem_type})")
    dims = []
    for d in t.shape.dim:
        if d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append(str(d.dim_value))
    return dtype, "[" + ", ".join(dims) + "]"


def main():
    parser = argparse.ArgumentParser(description="Inspect ONNX model inputs/outputs.")
    parser.add_argument("onnx", help="Path to .onnx file")
    args = parser.parse_args()

    try:
        import onnx
    except ImportError:
        print("pip install onnx", file=sys.stderr)
        return 1

    model = onnx.load(args.onnx)

    print(f"=== MODEL ===")
    print(f"  IR version:    {model.ir_version}")
    print(f"  Opset:         {', '.join(str(op.version) for op in model.opset_import)}")
    print(f"  Producer:      {model.producer_name} {model.producer_version}")
    if model.domain:
        print(f"  Domain:        {model.domain}")
    print(f"  Graph:         {model.graph.name}")
    print(f"  Nodes:         {len(model.graph.node)}")

    if model.metadata_props:
        print("=== METADATA ===")
        for prop in model.metadata_props:
            val = prop.value
            if len(val) > 120:
                val = val[:120] + "..."
            print(f"  {prop.key:30s}  {val}")

    print("=== INPUTS ===")
    for inp in model.graph.input:
        dtype, shape = describe_tensor(inp)
        print(f"  {inp.name:30s}  {dtype:10s}  {shape}")

    print("=== OUTPUTS ===")
    for out in model.graph.output:
        dtype, shape = describe_tensor(out)
        print(f"  {out.name:30s}  {dtype:10s}  {shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
