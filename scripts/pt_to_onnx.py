#!/usr/bin/env python3
"""Convert a policy .pt file to ONNX.

Supports:
  - TorchScript archives (load via torch.jit.load)
  - Training checkpoints with actor state_dict (load via torch.load + rebuild MLP)

This repo's C++ runner expects policies with:
  - input:  float32 tensor of shape [1, input_dim]
  - output: float32 tensor of shape [1, output_dim]

Typical usage:
  python3 scripts/pt_to_onnx.py \
    --pt src/legged_rl_deploy/policies/go2_velocity_unitree.pt \
    --onnx /tmp/go2_velocity_unitree.onnx \
    --input-dim 45 --output-dim 12 --opset 17 --check

Or read dims from a deploy config:
  python3 scripts/pt_to_onnx.py \
    --pt src/legged_rl_deploy/policies/go2_velocity_unitree.pt \
    --onnx /tmp/go2_velocity_unitree.onnx \
    --config src/legged_rl_deploy/config/go2-velocity-unitree.yaml --check
"""

from __future__ import annotations

import argparse
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PolicyDims:
    input_dim: int
    output_dim: int


def _load_dims_from_yaml(path: str) -> PolicyDims:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required for --config. Install with: python3 -m pip install pyyaml"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    policy = cfg.get("policy") if isinstance(cfg, dict) else None
    if not isinstance(policy, dict):
        raise RuntimeError(f"Invalid config YAML, missing 'policy': {path}")

    input_dim = policy.get("input_dim")
    output_dim = policy.get("output_dim")
    if not isinstance(input_dim, int) or not isinstance(output_dim, int):
        raise RuntimeError(f"Config YAML missing integer policy.input_dim/output_dim: {path}")

    return PolicyDims(input_dim=input_dim, output_dim=output_dim)


def _ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _activation_ctor(torch, name: str):
    n = name.lower()
    if n == "elu":
        return torch.nn.ELU
    if n == "relu":
        return torch.nn.ReLU
    if n == "tanh":
        return torch.nn.Tanh
    if n == "silu":
        return torch.nn.SiLU
    if n == "gelu":
        return torch.nn.GELU
    raise RuntimeError(
        f"Unsupported --checkpoint-activation '{name}'. "
        "Use one of: elu,relu,tanh,silu,gelu"
    )


def _build_policy_from_checkpoint(
    payload: object,
    dims: PolicyDims,
    checkpoint_activation: str,
):
    """Reconstruct a feedforward actor policy from a training checkpoint."""
    import torch

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Loaded .pt is not TorchScript, and torch.load() returned a non-dict object. "
            "This converter currently supports TorchScript or dict checkpoints."
        )

    state_dict = None
    for k in ("model_state_dict", "state_dict", "actor_state_dict", "policy_state_dict"):
        v = payload.get(k)
        if isinstance(v, dict):
            state_dict = v
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        sample_keys = list(payload.keys())[:20]
        raise RuntimeError(
            "Could not find a supported state_dict in checkpoint. "
            f"Top-level keys: {sample_keys}"
        )

    actor_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k, v in state_dict.items():
        if torch.is_tensor(v) and isinstance(k, str) and k.startswith("actor."):
            actor_state[k[len("actor.") :]] = v

    # Some checkpoints may already store actor-only keys as "0.weight", ...
    if not actor_state:
        for k, v in state_dict.items():
            if torch.is_tensor(v) and isinstance(k, str) and re.fullmatch(r"\d+\.(weight|bias)", k):
                actor_state[k] = v

    if not actor_state:
        sample_actor_like = [k for k in state_dict.keys() if isinstance(k, str)][:20]
        raise RuntimeError(
            "Checkpoint loaded, but could not locate actor weights. "
            f"Sample state_dict keys: {sample_actor_like}"
        )

    layer_ids: list[int] = []
    for k in actor_state.keys():
        m = re.fullmatch(r"(\d+)\.weight", k)
        if m and f"{m.group(1)}.bias" in actor_state:
            layer_ids.append(int(m.group(1)))
    layer_ids = sorted(set(layer_ids))
    if not layer_ids:
        raise RuntimeError(
            "Actor state_dict does not contain linear layer pairs '<idx>.weight' and '<idx>.bias'."
        )

    first_w = actor_state[f"{layer_ids[0]}.weight"]
    last_w = actor_state[f"{layer_ids[-1]}.weight"]
    if first_w.ndim != 2 or last_w.ndim != 2:
        raise RuntimeError("Unsupported actor layer shape: expected 2D linear weights.")

    inferred_input_dim = int(first_w.shape[1])
    inferred_output_dim = int(last_w.shape[0])

    if dims.input_dim != inferred_input_dim or dims.output_dim != inferred_output_dim:
        raise RuntimeError(
            "Config dimensions do not match checkpoint actor dimensions. "
            f"config=({dims.input_dim},{dims.output_dim}), "
            f"checkpoint=({inferred_input_dim},{inferred_output_dim})"
        )

    act = _activation_ctor(torch, checkpoint_activation)
    modules: list[tuple[str, torch.nn.Module]] = []
    for i, layer_id in enumerate(layer_ids):
        w = actor_state[f"{layer_id}.weight"]
        b = actor_state[f"{layer_id}.bias"]
        if w.ndim != 2 or b.ndim != 1 or int(w.shape[0]) != int(b.shape[0]):
            raise RuntimeError(
                f"Invalid linear params for layer {layer_id}: weight={tuple(w.shape)}, bias={tuple(b.shape)}"
            )
        modules.append((str(layer_id), torch.nn.Linear(int(w.shape[1]), int(w.shape[0]))))
        if i != len(layer_ids) - 1:
            modules.append((f"act_{layer_id}", act()))

    actor = torch.nn.Sequential(OrderedDict(modules))
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()

    mean = state_dict.get("actor_obs_normalizer._mean")
    std = state_dict.get("actor_obs_normalizer._std")
    if std is None and torch.is_tensor(state_dict.get("actor_obs_normalizer._var")):
        std = torch.sqrt(state_dict["actor_obs_normalizer._var"] + 1.0e-8)

    use_normalizer = torch.is_tensor(mean) and torch.is_tensor(std)
    obs_mean = None
    obs_std = None
    if use_normalizer:
        assert torch.is_tensor(mean)
        assert torch.is_tensor(std)
        obs_mean = mean.to(dtype=torch.float32).reshape(1, -1)
        obs_std = std.to(dtype=torch.float32).reshape(1, -1)
        if obs_mean.shape[-1] != inferred_input_dim or obs_std.shape[-1] != inferred_input_dim:
            raise RuntimeError(
                "Checkpoint actor_obs_normalizer shape mismatch with actor input dim. "
                f"mean={tuple(obs_mean.shape)}, std={tuple(obs_std.shape)}, input_dim={inferred_input_dim}"
            )

    class _CheckpointPolicy(torch.nn.Module):
        def __init__(self, actor_module, mean_tensor, std_tensor):
            super().__init__()
            self.actor = actor_module
            self._use_normalizer = (
                mean_tensor is not None
                and std_tensor is not None
                and torch.is_tensor(mean_tensor)
                and torch.is_tensor(std_tensor)
            )
            if self._use_normalizer:
                self.register_buffer("_obs_mean", mean_tensor)
                self.register_buffer("_obs_std", std_tensor)

        def forward(self, obs):
            x = obs
            if self._use_normalizer:
                x = (x - self._obs_mean) / torch.clamp(self._obs_std, min=1.0e-6)
            out = self.actor(x)
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

    return _CheckpointPolicy(actor, obs_mean, obs_std)


def _load_policy_module(
    pt_path: str,
    dims: PolicyDims,
    checkpoint_activation: str,
):
    """Load a policy module from either TorchScript or checkpoint state_dict."""
    import torch

    class _Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, obs):
            out = self.module(obs)
            # Some policies return (action, ...). Keep action.
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

    ts_err = None
    try:
        ts = torch.jit.load(pt_path, map_location="cpu")
        ts.eval()
        return _Wrapper(ts), "torchscript"
    except Exception as e:
        ts_err = e

    try:
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as ckpt_err:
        raise RuntimeError(
            "Failed to load .pt as TorchScript and as checkpoint.\n"
            f"torch.jit.load error: {ts_err}\n"
            f"torch.load error: {ckpt_err}"
        ) from ckpt_err

    policy = _build_policy_from_checkpoint(
        payload=payload,
        dims=dims,
        checkpoint_activation=checkpoint_activation,
    )
    policy.eval()
    return policy, "checkpoint"


def _export_torchscript_to_onnx(
    pt_path: str,
    onnx_path: str,
    dims: PolicyDims,
    opset: int,
    dynamic_batch: bool,
    checkpoint_activation: str,
) -> None:
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch (python package) is required for conversion. Install a CPU build with:\n"
            "  python3 -m pip install --upgrade pip\n"
            "  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "If you already use a conda/venv for training, run this script inside that env."
        ) from e

    if not os.path.exists(pt_path):
        raise FileNotFoundError(pt_path)

    policy_module, source = _load_policy_module(
        pt_path=pt_path,
        dims=dims,
        checkpoint_activation=checkpoint_activation,
    )
    policy_module.eval()
    print(f"Loaded policy source: {source}")

    dummy = torch.randn(1, dims.input_dim, dtype=torch.float32)

    # NOTE: Some TorchScript modules (including policies exported from IsaacLab/RSL-RL)
    # can trigger a legacy exporter error:
    #   "Tried to trace <...> but it is not part of the active trace"
    # Tracing a small wrapper first produces a stable TorchScript graph that the
    # ONNX exporter can consume.
    traced = torch.jit.trace(policy_module, dummy, strict=False)
    traced.eval()

    input_names = ["obs"]
    output_names = ["action"]
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"obs": {0: "batch"}, "action": {0: "batch"}}

    _ensure_parent_dir(onnx_path)

    with torch.no_grad():
        export_kwargs = dict(
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        try:
            # Newer PyTorch defaults to a dynamo-based exporter that can fail on some
            # TorchScript modules. Prefer the legacy exporter for robustness.
            torch.onnx.export(traced, dummy, onnx_path, dynamo=False, **export_kwargs)
        except TypeError:
            # Older PyTorch doesn't have the 'dynamo' flag.
            torch.onnx.export(traced, dummy, onnx_path, **export_kwargs)


def _check_onnx(
    onnx_path: str,
    pt_path: str,
    dims: PolicyDims,
    n_trials: int,
    atol: float,
    checkpoint_activation: str,
) -> None:
    try:
        import onnx
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "onnx is required for --check. Install with: python3 -m pip install onnx"
        ) from e

    try:
        import onnxruntime as ort
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "onnxruntime is required for --check. Install with: python3 -m pip install onnxruntime"
        ) from e

    import numpy as np

    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)

    # Prepare Torch runner
    import torch

    policy_module, source = _load_policy_module(
        pt_path=pt_path,
        dims=dims,
        checkpoint_activation=checkpoint_activation,
    )
    policy_module.eval()

    def torch_forward(obs_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs = torch.from_numpy(obs_np).to(dtype=torch.float32)
            out = policy_module(obs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out.to("cpu", torch.float32).contiguous()
            return out.numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    for i in range(n_trials):
        obs_np = np.random.randn(1, dims.input_dim).astype(np.float32)

        y_torch = torch_forward(obs_np)
        y_onnx = sess.run(["action"], {"obs": obs_np})[0]

        if y_torch.shape != (1, dims.output_dim):
            raise RuntimeError(f"Torch output shape unexpected: {y_torch.shape} (expected [1,{dims.output_dim}])")
        if y_onnx.shape != (1, dims.output_dim):
            raise RuntimeError(f"ONNX output shape unexpected: {y_onnx.shape} (expected [1,{dims.output_dim}])")

        max_abs = float(np.max(np.abs(y_torch - y_onnx)))
        if max_abs > atol:
            raise RuntimeError(
                f"ONNX check failed on trial {i}: max_abs_diff={max_abs} > atol={atol}"
            )

    print(f"ONNX check OK ({n_trials} trials, atol={atol}, source={source})")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Convert policy (.pt) to ONNX.")
    p.add_argument("--pt", required=True, help="Input policy file (.pt): TorchScript or checkpoint")
    p.add_argument("--onnx", required=True, help="Output ONNX file (.onnx)")
    p.add_argument("--config", help="Deploy YAML config to read policy.input_dim/output_dim")
    p.add_argument("--input-dim", type=int, help="Policy input dim (if not using --config)")
    p.add_argument("--output-dim", type=int, help="Policy output dim (if not using --config)")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    p.add_argument(
        "--checkpoint-activation",
        default="elu",
        help=(
            "Activation used when --pt is a training checkpoint (not TorchScript). "
            "Default: elu"
        ),
    )
    p.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with dynamic batch axis (obs/action dim0)",
    )
    p.add_argument("--check", action="store_true", help="Run onnx checker + compare outputs")
    p.add_argument("--check-trials", type=int, default=3, help="Number of random trials for --check")
    p.add_argument("--check-atol", type=float, default=1e-4, help="Abs diff tolerance for --check")
    args = p.parse_args(argv)

    if args.config:
        dims = _load_dims_from_yaml(args.config)
    else:
        if args.input_dim is None or args.output_dim is None:
            p.error("Provide --config or both --input-dim and --output-dim")
        dims = PolicyDims(input_dim=args.input_dim, output_dim=args.output_dim)

    _export_torchscript_to_onnx(
        pt_path=args.pt,
        onnx_path=args.onnx,
        dims=dims,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        checkpoint_activation=args.checkpoint_activation,
    )

    if args.check:
        _check_onnx(
            onnx_path=args.onnx,
            pt_path=args.pt,
            dims=dims,
            n_trials=args.check_trials,
            atol=args.check_atol,
            checkpoint_activation=args.checkpoint_activation,
        )

    print(f"Wrote: {args.onnx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
