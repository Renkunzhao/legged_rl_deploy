#!/usr/bin/env python3
"""Analyze 500 Hz Sim2Sim traces against physical Go2 motor T-omega envelopes.

Reads ``traces/<skill>.jsonl`` produced by ``eval_flip_mjlab_sim2sim.py`` and
the calibrated motor parameters from ``doc/actuator/actuator_verification.json``,
computes exact 4-quadrant ``dc_motor_clip`` and Q1/Q3 FOC envelope compliance,
and outputs:
  - ``metrics.json``
  - ``fig1_joint_groups_tn_scatter.png``
  - ``fig2_7skills_calf_tn_scatter.png``
  - ``tn_interactive.html``
  - ``README.md``
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

ACTUATOR_VERIF_JSON = Path(
    "/home/rkz/code/unitree_rl_mjlab-trampoline/doc/actuator/actuator_verification.json"
)
SKILLS = (
    "back_flip",
    "front_flip",
    "left_flip",
    "right_flip",
    "jump_up",
    "jump_forward",
    "jump_back",
)
SKILL_LABELS = {
    "back_flip": "后空翻 (back_flip)",
    "front_flip": "前空翻 (front_flip)",
    "left_flip": "左空翻 (left_flip)",
    "right_flip": "右空翻 (right_flip)",
    "jump_up": "原地高跳 (jump_up)",
    "jump_forward": "前跳 0.7m (jump_forward)",
    "jump_back": "后跳 -0.5m (jump_back)",
}
FLIP_AXES = {
    "back_flip": (np.array([0.0, -1.0, 0.0]), 360.0),
    "front_flip": (np.array([0.0, 1.0, 0.0]), 360.0),
    "left_flip": (np.array([-1.0, 0.0, 0.0]), 360.0),
    "right_flip": (np.array([1.0, 0.0, 0.0]), 360.0),
}
JOINT_GROUPS = {
    "hip": {"indices": [0, 3, 6, 9], "param_key": "hip_thigh", "kt": 0.63895, "title": "髋外展 Hip (6.33:1)"},
    "thigh": {"indices": [1, 4, 7, 10], "param_key": "hip_thigh", "kt": 0.63895, "title": "大腿 Thigh (6.33:1)"},
    "calf": {"indices": [2, 5, 8, 11], "param_key": "calf", "kt": 1.2248, "title": "小腿/膝 Calf (12.134:1)"},
}
PHASE_ORDER = ("Push-off", "Flight", "Landing", "Post-stand")
PHASE_COLORS = {
    "Push-off": "#d95f02",
    "Flight": "#1b6ca8",
    "Landing": "#7209b7",
    "Post-stand": "#6c757d",
}


def configure_cjk_matplotlib() -> str:
    """Select an available CJK font so rendered PNGs contain no tofu glyphs."""
    preferred = [
        "Noto Sans CJK SC",
        "Noto Sans CJK HK",
        "Noto Serif CJK SC",
        "AR PL UMing CN",
        "AR PL KaitiM GB",
        "Droid Sans Fallback",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in preferred if name in available), None)
    if chosen is None:
        raise RuntimeError(f"No CJK font found among {preferred}")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [chosen, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return chosen


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert [w, x, y, z] quaternion to 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def dc_clip_bounds(
    dq: np.ndarray, tau_pk: float, sat_28p8: float, w0_28p8: float, v_bus: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Exact 4-quadrant dc_motor_clip upper/lower bounds at bus voltage v_bus."""
    sc = v_bus / 28.8
    tau_st = sat_28p8 * sc
    w0 = w0_28p8 * sc
    w_sat = w0 * (1.0 + tau_pk / tau_st)
    v_c = np.clip(dq, -w_sat, w_sat)
    t_max = np.minimum(tau_pk, tau_st * (1.0 - v_c / w0))
    t_min = np.maximum(-tau_pk, tau_st * (-1.0 - v_c / w0))
    return t_min, t_max, w0, w_sat


def foc_motoring_cap(
    dq: np.ndarray, tau_pk: float, sat_28p8: float, w0_28p8: float, v_bus: float
) -> np.ndarray:
    """Q1/Q3 motoring torque cap |tau| <= min(tau_peak, max(0, tau_stall*(1 - |dq|/w0)))."""
    sc = v_bus / 28.8
    tau_st = sat_28p8 * sc
    w0 = w0_28p8 * sc
    return np.minimum(tau_pk, np.maximum(0.0, tau_st * (1.0 - np.abs(dq) / w0)))


def segment_phases(t_rel: np.ndarray, foot_force: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment 500 Hz samples into Push-off, Flight, Landing, and Post-stand."""
    ff_sum = foot_force.sum(axis=1)
    air_mask = ff_sum <= 5.0
    best_span: tuple[int, int] | None = None
    i = 0
    n = len(t_rel)
    while i < n:
        if not air_mask[i]:
            i += 1
            continue
        j = i
        while j < n and air_mask[j]:
            j += 1
        if (t_rel[j - 1] - t_rel[i]) >= 0.06 and t_rel[i] < 2.8:
            if best_span is None or (j - i) > (best_span[1] - best_span[0]):
                best_span = (i, j)
        i = j

    phases = np.full(n, "Post-stand", dtype=object)
    if best_span is None:
        phases[t_rel < 2.5] = "Push-off"
        return phases, {"has_flight": False, "t_takeoff": None, "t_landing": None, "flight_s": 0.0}

    i0, i1 = best_span
    t_takeoff = float(t_rel[i0])
    t_landing = float(t_rel[min(i1, n - 1)])
    phases[t_rel < t_takeoff] = "Push-off"
    phases[(t_rel >= t_takeoff) & (t_rel <= t_landing)] = "Flight"
    phases[(t_rel > t_landing) & (t_rel <= t_landing + 0.5)] = "Landing"
    phases[t_rel > t_landing + 0.5] = "Post-stand"
    return phases, {
        "has_flight": True,
        "t_takeoff": round(t_takeoff, 3),
        "t_landing": round(t_landing, 3),
        "flight_s": round(t_landing - t_takeoff, 3),
    }


def compute_flip_angle_deg(quats: np.ndarray, t_rel: np.ndarray, flight_info: dict[str, Any], axis_body: np.ndarray) -> float | None:
    """Integrate projected body-up angle about the world-locked flip axis."""
    if not flight_info["has_flight"]:
        return None
    t_to = flight_info["t_takeoff"]
    t_la = flight_info["t_landing"]
    idx0 = int(np.searchsorted(t_rel, max(0.0, t_to - 0.15)))
    R0 = quat_to_rot(quats[idx0])
    yaw0 = math.atan2(float(R0[1, 0]), float(R0[0, 0]))
    cy, sy = math.cos(yaw0), math.sin(yaw0)
    axis = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]) @ axis_body
    axis /= np.linalg.norm(axis)

    total = 0.0
    prev_up = R0[:, 2] - axis * float(np.dot(R0[:, 2], axis))
    prev_up /= max(1e-9, float(np.linalg.norm(prev_up)))
    for k in range(idx0 + 1, len(t_rel)):
        if t_rel[k] > t_la + 0.15:
            break
        up = quat_to_rot(quats[k])[:, 2]
        proj = up - axis * float(np.dot(up, axis))
        nrm = float(np.linalg.norm(proj))
        if nrm < 1e-3:
            continue
        curr = proj / nrm
        total += math.atan2(float(np.dot(axis, np.cross(prev_up, curr))), float(np.dot(prev_up, curr)))
        prev_up = curr
    return float(np.degrees(total))


def compute_group_stats(
    dq: np.ndarray,
    tau: np.ndarray,
    tau_pk: float,
    sat_28p8: float,
    w0_28p8: float,
    is_calf: bool,
    eps: float = 0.05,
) -> dict[str, Any]:
    """Compute 4-quadrant, Q1/Q3 motoring, and overspeed compliance metrics."""
    n = int(dq.size)
    if n == 0:
        return {"samples": 0}
    abs_dq = np.abs(dq)
    abs_tau = np.abs(tau)
    mot_mask = (tau * dq) > 0.0
    regen_mask = (tau * dq) < 0.0

    res: dict[str, Any] = {
        "samples": n,
        "max_abs_dq": round(float(abs_dq.max()), 3),
        "p99_abs_dq": round(float(np.percentile(abs_dq, 99)), 3),
        "max_abs_tau": round(float(abs_tau.max()), 3),
        "box_sat_pct": round(float(np.mean(abs_tau >= tau_pk - 0.10) * 100.0), 3),
        "max_mech_power_W": round(float(np.max(tau * dq)), 1),
        "min_regen_power_W": round(float(np.min(tau * dq)), 1),
    }

    for v_tag, v_bus in [("25p5V", 25.5), ("28p8V", 28.8), ("33p6V", 33.6)]:
        t_min, t_max, w0, w_sat = dc_clip_bounds(dq, tau_pk, sat_28p8, w0_28p8, v_bus)
        viol_4q = (tau > t_max + eps) | (tau < t_min - eps)
        excess_4q = np.maximum(0.0, np.maximum(tau - t_max, t_min - tau))

        cap_foc = foc_motoring_cap(dq, tau_pk, sat_28p8, w0_28p8, v_bus)
        viol_mot = mot_mask & (abs_tau > cap_foc + eps)
        excess_mot = np.where(mot_mask, np.maximum(0.0, abs_tau - cap_foc), 0.0)

        over_w0 = abs_dq > w0
        # Inside overspeed, cases where motor is still pushing (motoring) or zero/low braking
        over_w0_mot = over_w0 & mot_mask
        over_w0_regen = over_w0 & regen_mask

        res[v_tag] = {
            "w0": round(w0, 2),
            "w_sat": round(w_sat, 2),
            "viol_4q_pct": round(float(np.mean(viol_4q) * 100.0), 3),
            "viol_4q_count": int(np.sum(viol_4q)),
            "max_excess_4q_Nm": round(float(excess_4q.max()), 3),
            "viol_mot_pct": round(float(np.mean(viol_mot) * 100.0), 3),
            "max_excess_mot_Nm": round(float(excess_mot.max()), 3),
            "over_w0_pct": round(float(np.mean(over_w0) * 100.0), 3),
            "over_w0_mot_pct": round(float(np.mean(over_w0_mot) * 100.0), 3),
            "over_w0_regen_pct": round(float(np.mean(over_w0_regen) * 100.0), 3),
        }

    if is_calf:
        # Also evaluate against hw_trampoline_20261121 pronk anchor w0_eff = 21.5 rad/s @ 28.4 V
        over_21p5 = abs_dq > 21.5
        cap_21p5 = np.minimum(tau_pk, np.maximum(0.0, (127.15 * (21.5 / 18.07)) * (1.0 - abs_dq / 21.5)))
        viol_mot_21p5 = mot_mask & (abs_tau > cap_21p5 + eps)
        res["hw_pronk_21p5"] = {
            "over_21p5_pct": round(float(np.mean(over_21p5) * 100.0), 3),
            "viol_mot_21p5_pct": round(float(np.mean(viol_mot_21p5) * 100.0), 3),
        }
        res["over_wsat_24p53_pct"] = round(float(np.mean(abs_dq > 24.53) * 100.0), 3)

    return res


def load_skill_trace(trace_path: Path) -> dict[str, np.ndarray]:
    """Load 500 Hz JSONL trace into vectorized NumPy arrays."""
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    t_rel = np.array([r["t_rel"] for r in rows], dtype=float)
    q = np.array([r["q"] for r in rows], dtype=float)
    dq = np.array([r["dq"] for r in rows], dtype=float)
    tau_est = np.array([r["tau_est"] for r in rows], dtype=float)
    tau_pd_raw = np.array([r.get("tau_pd_raw", r["tau_est"]) for r in rows], dtype=float)
    tau_cmd = np.array([r.get("tau_cmd_applied", r["tau_est"]) for r in rows], dtype=float)
    foot_force = np.array([r["foot_force"] for r in rows], dtype=float)
    quat = np.array([r["quat"] for r in rows], dtype=float)
    pos = np.array([r.get("position", [0.0, 0.0, 0.0]) for r in rows], dtype=float)
    body_h = np.array([r.get("body_height", 0.0) for r in rows], dtype=float)
    safety = np.array([bool(r.get("safety_ok", True)) for r in rows], dtype=bool)
    return {
        "t_rel": t_rel,
        "q": q,
        "dq": dq,
        "tau_est": tau_est,
        "tau_pd_raw": tau_pd_raw,
        "tau_cmd": tau_cmd,
        "foot_force": foot_force,
        "quat": quat,
        "pos": pos,
        "body_h": body_h,
        "safety": safety,
    }


def draw_envelopes(ax: plt.Axes, params: dict[str, float], w_lim: float, show_legend: bool = False, is_calf: bool = False) -> None:
    """Plot 4-quadrant T-omega envelopes and XML box limit."""
    w = np.linspace(-w_lim, w_lim, 600)
    tau_pk = params["effort_limit"]
    sat = params["saturation_effort"]
    w0_nom = params["velocity_limit"]

    # Shaded feasible 28.8 V 4-quadrant region
    t_min_28, t_max_28, _, _ = dc_clip_bounds(w, tau_pk, sat, w0_nom, 28.8)
    ax.fill_between(w, t_min_28, t_max_28, color="#4ea8de", alpha=0.09, zorder=1)
    ax.axhline(0.0, color="#666666", lw=0.7, zorder=2)
    ax.axvline(0.0, color="#666666", lw=0.7, zorder=2)

    for v_bus, lbl, col, ls, lw in [
        (25.5, "25.5 V (DR下界)", "#e76f51", "--", 1.5),
        (28.8, "28.8 V (标称4Q契约)", "#1b6ca8", "-", 2.1),
        (33.6, "33.6 V (满电上界)", "#2b9348", "-.", 1.4),
    ]:
        tmin, tmax, _, _ = dc_clip_bounds(w, tau_pk, sat, w0_nom, v_bus)
        ax.plot(w, tmax, color=col, ls=ls, lw=lw, label=lbl if show_legend else None, zorder=5)
        ax.plot(w, tmin, color=col, ls=ls, lw=lw, zorder=5)

    # go2.xml box limit
    ax.plot([-w_lim, w_lim], [tau_pk, tau_pk], color="#343a40", ls=":", lw=1.0,
            label=f"XML 矩形限幅 (±{tau_pk:.1f} N·m)" if show_legend else None, zorder=4)
    ax.plot([-w_lim, w_lim], [-tau_pk, -tau_pk], color="#343a40", ls=":", lw=1.0, zorder=4)

    if is_calf:
        ax.axvline(21.5, color="#c1121f", ls=":", lw=1.2, alpha=0.85,
                   label="hw pronk 实测空载 (~21.5 rad/s)" if show_legend else None, zorder=4)
        ax.axvline(-21.5, color="#c1121f", ls=":", lw=1.2, alpha=0.85, zorder=4)


def plot_fig1(all_traces: dict[str, dict[str, Any]], phys_params: dict[str, Any], out_path: Path) -> None:
    """3-panel aggregated (dq, tau) scatter for Hip, Thigh, and Calf across all 7 skills."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), dpi=220)
    configs = [
        ("hip", 48.0, 42.0),
        ("thigh", 48.0, 42.0),
        ("calf", 52.0, 75.0),
    ]
    for idx, (jgrp, w_lim, t_lim) in enumerate(configs):
        ax = axes[idx]
        ginfo = JOINT_GROUPS[jgrp]
        params = phys_params[ginfo["param_key"]]
        j_idx = ginfo["indices"]
        draw_envelopes(ax, params, w_lim, show_legend=(idx == 2), is_calf=(jgrp == "calf"))

        # Plot faint unclipped PD demand first so tau_est stays foreground
        dq_all = np.concatenate([all_traces[s]["dq"][:, j_idx].ravel() for s in SKILLS])
        pd_all = np.concatenate([all_traces[s]["tau_pd_raw"][:, j_idx].ravel() for s in SKILLS])
        est_all = np.concatenate([all_traces[s]["tau_est"][:, j_idx].ravel() for s in SKILLS])
        ph_all = np.concatenate([np.repeat(all_traces[s]["phases"], 4) for s in SKILLS])

        ax.scatter(dq_all, pd_all, s=3.0, color="#adb5bd", alpha=0.14, rasterized=True,
                   label="未裁剪 PD 需求 (tau_pd_raw)" if idx == 2 else None, zorder=2)

        for ph in ("Post-stand", "Flight", "Landing", "Push-off"):
            m = ph_all == ph
            if np.any(m):
                ax.scatter(
                    dq_all[m], est_all[m],
                    s=5.0 if ph != "Post-stand" else 3.0,
                    color=PHASE_COLORS[ph],
                    alpha=0.45 if ph != "Post-stand" else 0.18,
                    rasterized=True,
                    label=f"实施力矩 tau_est [{ph}]" if idx == 2 else None,
                    zorder=3 if ph == "Post-stand" else 4,
                )

        ax.set_xlim(-w_lim, w_lim)
        ax.set_ylim(-t_lim, t_lim)
        ax.set_xlabel("关节角速度 dq (rad/s)")
        ax.set_ylabel("关节力矩 τ (N·m)")
        ax.set_title(f"({chr(97 + idx)}) {ginfo['title']}", fontweight="bold")
        ax.grid(True, ls=":", alpha=0.35)

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True,
               facecolor="white", framealpha=0.95, bbox_to_anchor=(0.5, 0.99), fontsize=8.4)
    fig.suptitle(
        "My_unitree_go2_gym (origin/flip_mjlab) 7技能 2ms Sim2Sim 关节力矩-转速散点 vs 实物电机包络",
        fontsize=12.0, fontweight="bold", y=1.03,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.89])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig2(
    all_traces: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    phys_params: dict[str, Any],
    out_path: Path,
) -> None:
    """7-panel per-skill Calf T-omega scatter plots."""
    fig, axes = plt.subplots(2, 4, figsize=(17.2, 8.6), dpi=220)
    flat_axes = axes.ravel()
    calf_params = phys_params["calf"]
    j_idx = JOINT_GROUPS["calf"]["indices"]
    w_lim, t_lim = 50.0, 70.0

    for i, skill in enumerate(SKILLS):
        ax = flat_axes[i]
        tr = all_traces[skill]
        draw_envelopes(ax, calf_params, w_lim, show_legend=(i == 0), is_calf=True)
        dq = tr["dq"][:, j_idx].ravel()
        pd_raw = tr["tau_pd_raw"][:, j_idx].ravel()
        tau_est = tr["tau_est"][:, j_idx].ravel()
        ph = np.repeat(tr["phases"], 4)

        ax.scatter(dq, pd_raw, s=3.5, color="#adb5bd", alpha=0.18, rasterized=True,
                   label="未裁剪 PD 需求" if i == 0 else None, zorder=2)
        for p_name in ("Post-stand", "Flight", "Landing", "Push-off"):
            m = ph == p_name
            if np.any(m):
                ax.scatter(dq[m], tau_est[m], s=5.5, color=PHASE_COLORS[p_name],
                           alpha=0.50 if p_name != "Post-stand" else 0.20, rasterized=True,
                           label=f"tau_est [{p_name}]" if i == 0 else None, zorder=4)

        cm = metrics["skills"][skill]["joint_groups"]["calf"]["tau_est"]["ALL"]
        kin = metrics["skills"][skill]["kinematics"]
        badge = (
            f"28.8V 4Q越界: {cm['28p8V']['viol_4q_pct']:.1f}%\n"
            f"|dq|_max: {cm['max_abs_dq']:.1f} rad/s\n"
            f"超21.5: {cm['hw_pronk_21p5']['over_21p5_pct']:.1f}%\n"
            f"姿态: {'站稳' if kin['stayed_upright'] else '倾倒'}"
        )
        ax.text(
            0.03, 0.96, badge, transform=ax.transAxes, va="top", ha="left", fontsize=7.6,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#adb5bd", "alpha": 0.92},
            zorder=8,
        )
        ax.set_xlim(-w_lim, w_lim)
        ax.set_ylim(-t_lim, t_lim)
        ax.set_title(SKILL_LABELS[skill], fontweight="bold", fontsize=9.8)
        ax.set_xlabel("膝角速度 dq (rad/s)", fontsize=8.5)
        ax.set_ylabel("膝力矩 τ (N·m)", fontsize=8.5)
        ax.grid(True, ls=":", alpha=0.35)

    # Last panel holds the shared legend + global note
    ax_leg = flat_axes[7]
    ax_leg.axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc="center", frameon=True, facecolor="white", fontsize=8.8)

    fig.suptitle(
        "分技能膝关节 (Calf 12.134:1) 2ms Sim2Sim 力矩-转速散点与 GO-M8010-6 物理包络对比",
        fontsize=12.4, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_html_payload(all_traces: dict[str, dict[str, Any]], metrics: dict[str, Any], phys_params: dict[str, Any]) -> dict[str, Any]:
    """Pack metrics and compact representative scatter points for interactive HTML rendering."""
    skills_payload: dict[str, Any] = {}
    for skill in SKILLS:
        tr = all_traces[skill]
        phases = tr["phases"]
        grp_pts: dict[str, list[list[Any]]] = {}
        for grp_name, ginfo in [("hip_thigh", [0, 1, 3, 4, 6, 7, 9, 10]), ("calf", [2, 5, 8, 11])]:
            dq = tr["dq"][:, ginfo].ravel()
            tau_est = tr["tau_est"][:, ginfo].ravel()
            tau_pd = tr["tau_pd_raw"][:, ginfo].ravel()
            ph = np.repeat(phases, len(ginfo))

            # Keep ALL extreme/high-speed/high-torque points; thin low-load post-stand by 6x
            keep = (ph != "Post-stand") | (np.abs(dq) > 6.0) | (np.abs(tau_est) > 15.0)
            idx_keep = np.flatnonzero(keep)
            idx_bg = np.flatnonzero(~keep)[::6]
            sel = np.sort(np.concatenate([idx_keep, idx_bg]))
            # Further cap if huge while keeping all |dq|>12 or |tau|>20
            if len(sel) > 4500:
                hi = sel[(np.abs(dq[sel]) > 12.0) | (np.abs(tau_est[sel]) > 20.0)]
                lo = sel[~((np.abs(dq[sel]) > 12.0) | (np.abs(tau_est[sel]) > 20.0))][::2]
                sel = np.sort(np.concatenate([hi, lo]))
            pts = [
                [
                    round(float(dq[k]), 2),
                    round(float(tau_est[k]), 2),
                    round(float(np.clip(tau_pd[k], -160.0, 160.0)), 2),
                    PHASE_ORDER.index(str(ph[k])),
                ]
                for k in sel
            ]
            grp_pts[grp_name] = pts
        skills_payload[skill] = {
            "label": SKILL_LABELS[skill],
            "points": grp_pts,
            "summary": metrics["skills"][skill],
        }
    return {
        "skills": list(SKILLS),
        "phase_names": list(PHASE_ORDER),
        "phys_params": phys_params,
        "data": skills_payload,
    }


def write_interactive_html(payload: dict[str, Any], out_path: Path) -> None:
    """Write self-contained interactive HTML adhering to html_render style/theme rules."""
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Go2 7技能 Sim2Sim 力矩-转速分析</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 0;
    font-family: var(--font-sans, system-ui, -apple-system, sans-serif);
    color: var(--foreground);
    font-size: 13px;
    line-height: 1.45;
  }}
  .wrap {{
    width: 100%;
    max-width: 728px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  .box {{
    background: var(--card);
    color: var(--card-foreground);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
  }}
  .controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    justify-content: space-between;
  }}
  .btn-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }}
  button {{
    background: var(--secondary);
    color: var(--secondary-foreground, var(--foreground));
    border: 1px solid var(--border);
    border-radius: calc(var(--radius) - 2px);
    padding: 4px 9px;
    font-size: 12px;
    cursor: pointer;
    font-family: inherit;
  }}
  button.active {{
    background: var(--primary);
    color: var(--primary-foreground, #fff);
    border-color: var(--primary);
    font-weight: 600;
  }}
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
    margin-top: 10px;
  }}
  .kpi {{
    background: var(--muted);
    border-radius: calc(var(--radius) - 2px);
    padding: 8px 10px;
  }}
  .kpi .lbl {{
    font-size: 11px;
    color: var(--muted-foreground);
  }}
  .kpi .val {{
    font-size: 15px;
    font-weight: 700;
    font-family: var(--font-mono, monospace);
    margin-top: 2px;
  }}
  .charts {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }}
  @media (max-width: 600px) {{
    .charts {{ grid-template-columns: 1fr; }}
    .kpi-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  }}
  .chart-title {{
    font-size: 13px;
    font-weight: 600;
    margin: 0 0 6px 0;
  }}
  canvas {{
    display: block;
    width: 100%;
    height: 270px;
  }}
  canvas#barCanvas {{
    height: 210px;
  }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 8px;
    font-size: 11px;
    color: var(--muted-foreground);
  }}
  .leg-item {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }}
  .dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
  }}
  .line-sw {{
    width: 14px;
    height: 2px;
    display: inline-block;
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="box">
    <div class="controls">
      <div class="btn-row" id="skillBtns"></div>
      <div class="btn-row" id="modeBtns">
        <button data-mode="est" class="active">实施力矩 τ_est</button>
        <button data-mode="pd">未裁剪需求 τ_pd</button>
      </div>
    </div>
    <div class="kpi-row" id="kpiRow"></div>
  </div>

  <div class="charts">
    <div class="box">
      <div class="chart-title">髋 / 大腿 (6.33:1) T-ω 散点</div>
      <canvas id="hipCanvas"></canvas>
    </div>
    <div class="box">
      <div class="chart-title">小腿 / 膝关节 (12.134:1) T-ω 散点</div>
      <canvas id="calfCanvas"></canvas>
    </div>
  </div>

  <div class="box">
    <div class="chart-title">各技能膝关节 28.8 V 越界率 (%) 与峰值转速对比</div>
    <canvas id="barCanvas"></canvas>
    <div class="legend">
      <span class="leg-item"><span class="dot" style="background:#d95f02"></span>推蹬 Push-off</span>
      <span class="leg-item"><span class="dot" style="background:#1b6ca8"></span>腾空 Flight</span>
      <span class="leg-item"><span class="dot" style="background:#7209b7"></span>落地 Landing</span>
      <span class="leg-item"><span class="dot" style="background:#6c757d"></span>站立 Post-stand</span>
      <span class="leg-item"><span class="line-sw" style="background:#1b6ca8"></span>28.8V 4Q包络</span>
      <span class="leg-item"><span class="line-sw" style="border-top:2px dashed #e76f51"></span>25.5V DR包络</span>
      <span class="leg-item"><span class="line-sw" style="border-top:2px dotted #c1121f"></span>hw pronk 21.5 rad/s</span>
    </div>
  </div>
</div>

<script>
const PAYLOAD = {data_json};
let selectedSkill = "ALL";
let torqueMode = "est"; // "est" | "pd"
const PHASE_COLORS = ["#d95f02", "#1b6ca8", "#7209b7", "#6c757d"];

function cssVar(name, fallback) {{
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}}

function dcBounds(dq, tauPk, sat28, w0_28, vBus) {{
  const sc = vBus / 28.8;
  const tauSt = sat28 * sc;
  const w0 = w0_28 * sc;
  const wSat = w0 * (1.0 + tauPk / tauSt);
  const vc = Math.max(-wSat, Math.min(wSat, dq));
  return [
    Math.max(-tauPk, tauSt * (-1.0 - vc / w0)),
    Math.min(tauPk, tauSt * (1.0 - vc / w0)),
  ];
}}

function drawScatter(canvasId, grpKey, params, wLim, tLim, isCalf) {{
  const canvas = document.getElementById(canvasId);
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const W = Math.max(240, Math.round(rect.width));
  const H = 270;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const fg = cssVar("--foreground", "#e5e7eb");
  const muted = cssVar("--muted-foreground", "#9ca3af");
  const border = cssVar("--border", "#374151");

  const pad = {{ l: 38, r: 10, t: 10, b: 28 }};
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const sx = w => pad.l + ((w + wLim) / (2 * wLim)) * pw;
  const sy = t => pad.t + ((tLim - t) / (2 * tLim)) * ph;

  ctx.clearRect(0, 0, W, H);

  // Grid & axes
  ctx.strokeStyle = border;
  ctx.lineWidth = 0.7;
  ctx.fillStyle = muted;
  ctx.font = "10px monospace";
  for (const wv of [-40, -20, 0, 20, 40]) {{
    if (Math.abs(wv) > wLim) continue;
    const x = sx(wv);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
    ctx.textAlign = "center";
    ctx.fillText(String(wv), x, H - 12);
  }}
  const tTicks = isCalf ? [-60, -30, 0, 30, 60] : [-40, -20, 0, 20, 40];
  for (const tv of tTicks) {{
    if (Math.abs(tv) > tLim) continue;
    const y = sy(tv);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(String(tv), pad.l - 4, y + 3);
  }}
  ctx.fillStyle = muted;
  ctx.textAlign = "center";
  ctx.fillText("dq (rad/s)", pad.l + pw / 2, H - 2);

  // Envelopes
  const drawCurve = (vBus, color, dash) => {{
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = dash.length ? 1.4 : 2.0;
    ctx.setLineDash(dash);
    for (const boundIdx of [0, 1]) {{
      ctx.beginPath();
      for (let i = 0; i <= 120; i++) {{
        const w = -wLim + (2 * wLim * i) / 120;
        const b = dcBounds(w, params.effort_limit, params.saturation_effort, params.velocity_limit, vBus);
        const t = Math.max(-tLim, Math.min(tLim, b[boundIdx]));
        const x = sx(w), y = sy(t);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
      ctx.stroke();
    }}
    ctx.restore();
  }};
  drawCurve(25.5, "#e76f51", [4, 3]);
  drawCurve(28.8, "#1b6ca8", []);
  drawCurve(33.6, "#2b9348", [6, 3, 2, 3]);

  if (isCalf) {{
    ctx.save();
    ctx.strokeStyle = "#c1121f";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([2, 3]);
    for (const xv of [-21.5, 21.5]) {{
      const x = sx(xv);
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
    }}
    ctx.restore();
  }}

  // Scatter points
  const skills = selectedSkill === "ALL" ? PAYLOAD.skills : [selectedSkill];
  const tIdx = torqueMode === "est" ? 1 : 2;
  for (const sk of skills) {{
    const pts = PAYLOAD.data[sk].points[grpKey];
    for (let i = 0; i < pts.length; i++) {{
      const p = pts[i];
      const w = p[0], t = p[tIdx], phId = p[3];
      if (Math.abs(w) > wLim || Math.abs(t) > tLim) continue;
      ctx.fillStyle = PHASE_COLORS[phId];
      ctx.globalAlpha = phId === 3 ? 0.20 : 0.55;
      ctx.fillRect(sx(w) - 1.3, sy(t) - 1.3, 2.6, 2.6);
    }}
  }}
  ctx.globalAlpha = 1.0;
}}

function drawBars() {{
  const canvas = document.getElementById("barCanvas");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const W = Math.max(240, Math.round(rect.width));
  const H = 210;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const fg = cssVar("--foreground", "#e5e7eb");
  const muted = cssVar("--muted-foreground", "#9ca3af");
  const border = cssVar("--border", "#374151");

  const pad = {{ l: 92, r: 102, t: 10, b: 22 }};
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const n = PAYLOAD.skills.length;
  const rowH = ph / n;
  ctx.clearRect(0, 0, W, H);
  ctx.font = "11px monospace";

  const maxViol = 35.0;
  for (let i = 0; i < n; i++) {{
    const sk = PAYLOAD.skills[i];
    const cm = PAYLOAD.data[sk].summary.joint_groups.calf[torqueMode === "est" ? "tau_est" : "tau_pd_raw"].ALL;
    const v28 = cm["28p8V"].viol_4q_pct;
    const dqMax = cm.max_abs_dq;
    const y = pad.t + i * rowH;
    const bw = Math.min(pw, (v28 / maxViol) * pw);

    ctx.fillStyle = sk === selectedSkill ? cssVar("--primary", "#3b82f6") : fg;
    ctx.textAlign = "right";
    ctx.fillText(sk, pad.l - 8, y + rowH * 0.65);

    ctx.fillStyle = border;
    ctx.fillRect(pad.l, y + rowH * 0.22, pw, rowH * 0.56);

    ctx.fillStyle = v28 > 5.0 ? "#e76f51" : (v28 > 1.0 ? "#f4a261" : "#2b9348");
    ctx.fillRect(pad.l, y + rowH * 0.22, Math.max(2, bw), rowH * 0.56);

    ctx.fillStyle = muted;
    ctx.textAlign = "left";
    ctx.fillText(`${{v28.toFixed(1)}}% (${{dqMax.toFixed(1)}})`, pad.l + pw + 6, y + rowH * 0.65);
  }}
}}

function updateKPIs() {{
  const key = torqueMode === "est" ? "tau_est" : "tau_pd_raw";
  let violCalf28 = 0, maxDqCalf = 0, maxTauCalf = 0, maxDqHipThigh = 0;
  const skills = selectedSkill === "ALL" ? PAYLOAD.skills : [selectedSkill];
  let totalCalfN = 0, totalCalfV = 0;
  for (const sk of skills) {{
    const s = PAYLOAD.data[sk].summary.joint_groups;
    const c = s.calf[key].ALL;
    totalCalfN += c.samples;
    totalCalfV += c["28p8V"].viol_4q_count;
    maxDqCalf = Math.max(maxDqCalf, c.max_abs_dq);
    maxTauCalf = Math.max(maxTauCalf, c.max_abs_tau);
    maxDqHipThigh = Math.max(maxDqHipThigh, s.hip[key].ALL.max_abs_dq, s.thigh[key].ALL.max_abs_dq);
  }}
  violCalf28 = (totalCalfV / Math.max(1, totalCalfN)) * 100.0;
  let uprightText = "";
  if (selectedSkill === "ALL") {{
    const ok = PAYLOAD.skills.filter(sk => PAYLOAD.data[sk].summary.kinematics.stayed_upright).length;
    uprightText = `${{ok}} / ${{PAYLOAD.skills.length}} 站稳`;
  }} else {{
    const kin = PAYLOAD.data[selectedSkill].summary.kinematics;
    uprightText = `${{kin.stayed_upright ? "站稳" : "倾倒"}} (${{kin.end_tilt_deg.toFixed(0)}}°)`;
  }}

  const items = [
    ["小腿 28.8V 越界率", `${{violCalf28.toFixed(2)}}%`],
    ["小腿峰值转速 |dq|", `${{maxDqCalf.toFixed(1)}} rad/s`],
    ["髋/大腿峰值 |dq|", `${{maxDqHipThigh.toFixed(1)}} rad/s`],
    ["2ms Sim2Sim 落地", uprightText],
  ];
  document.getElementById("kpiRow").innerHTML = items.map(([l, v]) =>
    `<div class="kpi"><div class="lbl">${{l}}</div><div class="val">${{v}}</div></div>`
  ).join("");
}}

function renderAll() {{
  updateKPIs();
  drawScatter("hipCanvas", "hip_thigh", PAYLOAD.phys_params.hip_thigh, 48, torqueMode === "est" ? 32 : 85, false);
  drawScatter("calfCanvas", "calf", PAYLOAD.phys_params.calf, 50, torqueMode === "est" ? 55 : 140, true);
  drawBars();
}}

function init() {{
  const sRow = document.getElementById("skillBtns");
  const allBtn = document.createElement("button");
  allBtn.textContent = "全部 7 技能";
  allBtn.className = "active";
  allBtn.onclick = () => {{
    selectedSkill = "ALL";
    [...sRow.children].forEach(b => b.classList.remove("active"));
    allBtn.classList.add("active");
    renderAll();
  }};
  sRow.appendChild(allBtn);

  for (const sk of PAYLOAD.skills) {{
    const btn = document.createElement("button");
    btn.textContent = sk;
    btn.onclick = () => {{
      selectedSkill = sk;
      [...sRow.children].forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderAll();
    }};
    sRow.appendChild(btn);
  }}

  const mRow = document.getElementById("modeBtns");
  mRow.querySelectorAll("button").forEach(btn => {{
    btn.onclick = () => {{
      torqueMode = btn.dataset.mode;
      mRow.querySelectorAll("button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderAll();
    }};
  }});

  renderAll();
  window.addEventListener("resize", renderAll);
}}
init();
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def write_markdown_report(metrics: dict[str, Any], out_dir: Path) -> None:
    """Save Chinese quantitative evaluation report to README.md."""
    lines = [
        "# `My_unitree_go2_gym` (`origin/flip_mjlab`) 7 技能 2 ms Sim2Sim 力矩-转速特性核验",
        "",
        "本报告在标定 2 ms 刚性地面 `unitree_mujoco` (`scene.xml` + `go2.xml`, 16.514 kg) 闭环仿真中运行全部 7 个空翻/跳跃技能（各 4.0 s @ 500 Hz），对照 [go2_motor_characteristics_zh.tex](/home/rkz/code/unitree_rl_mjlab-trampoline/doc/actuator/go2_motor_characteristics_zh.tex) 与 [actuator_verification.json](/home/rkz/code/unitree_rl_mjlab-trampoline/doc/actuator/actuator_verification.json) 中的 `GO-M8010-6` / 8S 标称 `28.8 V` 四象限外特性包络做逐帧验证。",
        "",
        "## 1. 总览（2 ms Sim2Sim 落地表现与小腿瓶颈指标）",
        "",
        "| 技能 | 腾空时长 (s) | 机身最高 z (m) | 翻转角 (deg) | 末帧站立 | 膝 `max |dq|` (rad/s) | 膝 `28.8V` 4Q 越界 (`tau_est`) | 膝 `25.5V` 4Q 越界 (`tau_est`) | 膝 `>21.5 rad/s` (hw pronk) | 膝 `tau_pd_raw` 峰值 (N·m) | 估计峰值母线电流 (A) |",
        "|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sk in SKILLS:
        s = metrics["skills"][sk]
        k = s["kinematics"]
        c_est = s["joint_groups"]["calf"]["tau_est"]["ALL"]
        c_pd = s["joint_groups"]["calf"]["tau_pd_raw"]["ALL"]
        flip_s = f"{k['flip_angle_deg']:.1f}" if k["flip_angle_deg"] is not None else "—"
        upright_s = f"是 ({k['end_tilt_deg']:.1f}°)" if k["stayed_upright"] else f"否 ({k['end_tilt_deg']:.1f}°)"
        lines.append(
            f"| `{sk}` | {k['flight']['flight_s']:.3f} | {k['z_max_m']:.3f} | {flip_s} | {upright_s} | "
            f"{c_est['max_abs_dq']:.2f} | {c_est['28p8V']['viol_4q_pct']:.2f}% | {c_est['25p5V']['viol_4q_pct']:.2f}% | "
            f"{c_est['hw_pronk_21p5']['over_21p5_pct']:.2f}% | {c_pd['max_abs_tau']:.1f} | {s['peak_ibus_est_28p8V_A']:.1f} |"
        )

    lines.extend([
        "",
        "## 2. 全部 7 技能汇总（按关节组 × 动作阶段，`tau_est` 实际实施力矩）",
        "",
        "| 关节组 | 阶段 | 样本数 | `max |dq|` | `max |tau_est|` | `28.8V` 4Q 越界 | `28.8V` Q1/Q3 驱动越界 | `25.5V` 4Q 越界 | 超空载 `|dq|>ω0` (28.8V) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for grp in ("hip", "thigh", "calf"):
        for ph in ("ALL", *PHASE_ORDER):
            st = metrics["aggregate"][grp]["tau_est"].get(ph)
            if not st or st.get("samples", 0) == 0:
                continue
            lines.append(
                f"| `{grp}` | {ph} | {st['samples']} | {st['max_abs_dq']:.2f} | {st['max_abs_tau']:.2f} | "
                f"{st['28p8V']['viol_4q_pct']:.2f}% | {st['28p8V']['viol_mot_pct']:.2f}% | "
                f"{st['25p5V']['viol_4q_pct']:.2f}% | {st['28p8V']['over_w0_pct']:.2f}% |"
            )

    lines.extend([
        "",
        "## 3. 图表文件",
        "",
        "- [fig1_joint_groups_tn_scatter.png](./fig1_joint_groups_tn_scatter.png)",
        "- [fig2_7skills_calf_tn_scatter.png](./fig2_7skills_calf_tn_scatter.png)",
        "- [metrics.json](./metrics.json)",
        "",
    ])
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    configure_cjk_matplotlib()
    verif = json.loads(ACTUATOR_VERIF_JSON.read_text(encoding="utf-8"))
    phys_params = verif["deploy_contract"]["phys_actuator_params"]
    r_eq = float(verif["go_m8010_6"]["r_eq_ohm"])

    all_traces: dict[str, dict[str, Any]] = {}
    skills_metrics: dict[str, Any] = {}

    for skill in SKILLS:
        trace_path = args.results_dir / "traces" / f"{skill}.jsonl"
        tr = load_skill_trace(trace_path)
        phases, flight_info = segment_phases(tr["t_rel"], tr["foot_force"])
        tr["phases"] = phases
        all_traces[skill] = tr

        tilts_deg = np.array([math.degrees(math.acos(float(np.clip(quat_to_rot(q)[2, 2], -1.0, 1.0)))) for q in tr["quat"]])
        zs = tr["pos"][:, 2]
        disp_xy = float(np.linalg.norm(tr["pos"][-1, :2] - tr["pos"][0, :2]))
        flip_angle = (
            compute_flip_angle_deg(tr["quat"], tr["t_rel"], flight_info, FLIP_AXES[skill][0])
            if skill in FLIP_AXES
            else None
        )

        # Whole-robot bus current estimation (Eq. 4 of go2_motor_characteristics_zh.tex)
        kt_vec = np.array([0.63895, 0.63895, 1.2248] * 4)
        p_mech_pos = np.maximum(0.0, tr["tau_est"] * tr["dq"]) / 0.85
        p_cu = ((tr["tau_est"] / kt_vec[None, :]) ** 2) * r_eq
        p_elec = 120.0 + np.sum(p_mech_pos + p_cu, axis=1)
        ibus_est = p_elec / 28.8

        grp_metrics: dict[str, Any] = {}
        for grp, ginfo in JOINT_GROUPS.items():
            p = phys_params[ginfo["param_key"]]
            j_idx = ginfo["indices"]
            grp_metrics[grp] = {}
            for t_key in ("tau_est", "tau_pd_raw"):
                arr_dq = tr["dq"][:, j_idx]
                arr_t = tr[t_key][:, j_idx]
                phase_res = {
                    "ALL": compute_group_stats(
                        arr_dq.ravel(), arr_t.ravel(),
                        p["effort_limit"], p["saturation_effort"], p["velocity_limit"],
                        is_calf=(grp == "calf"),
                    )
                }
                for ph_name in PHASE_ORDER:
                    m = phases == ph_name
                    phase_res[ph_name] = compute_group_stats(
                        arr_dq[m].ravel(), arr_t[m].ravel(),
                        p["effort_limit"], p["saturation_effort"], p["velocity_limit"],
                        is_calf=(grp == "calf"),
                    )
                grp_metrics[grp][t_key] = phase_res

        skills_metrics[skill] = {
            "samples": int(len(tr["t_rel"])),
            "duration_s": round(float(tr["t_rel"][-1]), 3),
            "safety_ok": bool(np.all(tr["safety"])),
            "kinematics": {
                "flight": flight_info,
                "z_start_m": round(float(zs[0]), 3),
                "z_max_m": round(float(zs.max()), 3),
                "z_end_m": round(float(zs[-1]), 3),
                "disp_x_m": round(float(tr["pos"][-1, 0] - tr["pos"][0, 0]), 3),
                "disp_y_m": round(float(tr["pos"][-1, 1] - tr["pos"][0, 1]), 3),
                "disp_xy_m": round(disp_xy, 3),
                "max_tilt_deg": round(float(tilts_deg.max()), 2),
                "end_tilt_deg": round(float(tilts_deg[-1]), 2),
                "flip_angle_deg": round(flip_angle, 2) if flip_angle is not None else None,
                "stayed_upright": bool(tilts_deg[-1] < 30.0 and zs[-1] > 0.22 and np.all(tr["safety"])),
            },
            "peak_ibus_est_28p8V_A": round(float(ibus_est.max()), 1),
            "joint_groups": grp_metrics,
        }

    aggregate: dict[str, Any] = {}
    for grp, ginfo in JOINT_GROUPS.items():
        p = phys_params[ginfo["param_key"]]
        j_idx = ginfo["indices"]
        aggregate[grp] = {}
        dq_cat = np.concatenate([all_traces[s]["dq"][:, j_idx] for s in SKILLS], axis=0)
        ph_cat = np.concatenate([all_traces[s]["phases"] for s in SKILLS], axis=0)
        for t_key in ("tau_est", "tau_pd_raw"):
            t_cat = np.concatenate([all_traces[s][t_key][:, j_idx] for s in SKILLS], axis=0)
            phase_res = {
                "ALL": compute_group_stats(
                    dq_cat.ravel(), t_cat.ravel(),
                    p["effort_limit"], p["saturation_effort"], p["velocity_limit"],
                    is_calf=(grp == "calf"),
                )
            }
            for ph_name in PHASE_ORDER:
                m = ph_cat == ph_name
                phase_res[ph_name] = compute_group_stats(
                    dq_cat[m].ravel(), t_cat[m].ravel(),
                    p["effort_limit"], p["saturation_effort"], p["velocity_limit"],
                    is_calf=(grp == "calf"),
                )
            aggregate[grp][t_key] = phase_res

    metrics = {
        "actuator_reference": str(ACTUATOR_VERIF_JSON),
        "phys_actuator_params": phys_params,
        "skills": skills_metrics,
        "aggregate": aggregate,
    }
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_fig1(all_traces, phys_params, args.results_dir / "fig1_joint_groups_tn_scatter.png")
    plot_fig2(all_traces, metrics, phys_params, args.results_dir / "fig2_7skills_calf_tn_scatter.png")
    payload = build_html_payload(all_traces, metrics, phys_params)
    write_interactive_html(payload, args.results_dir / "tn_interactive.html")
    write_markdown_report(metrics, args.results_dir)
    print(f"[OK] Wrote metrics, PNGs, HTML, and README to {args.results_dir}")


if __name__ == "__main__":
    main()
