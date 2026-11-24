#!/usr/bin/env python3
"""Compare 2 ms Sim2Sim results before (box-only) vs after physical T-omega curve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from analyze_tn_envelopes import (  # noqa: E402
    ACTUATOR_VERIF_JSON,
    JOINT_GROUPS,
    PHASE_COLORS,
    PHASE_ORDER,
    SKILL_LABELS,
    SKILLS,
    dc_clip_bounds,
    draw_envelopes,
    load_skill_trace,
    segment_phases,
)


def configure_cjk_matplotlib() -> str:
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


def load_run_bundle(results_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    traces: dict[str, dict[str, Any]] = {}
    for sk in SKILLS:
        tr = load_skill_trace(results_dir / "traces" / f"{sk}.jsonl")
        phases, _ = segment_phases(tr["t_rel"], tr["foot_force"])
        tr["phases"] = phases
        traces[sk] = tr
    return metrics, traces


def build_comparison_summary(
    box_metrics: dict[str, Any],
    tn28_metrics: dict[str, Any],
    tn25_metrics: dict[str, Any],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for sk in SKILLS:
        row: dict[str, Any] = {}
        for label, m in [("box", box_metrics), ("tn28", tn28_metrics), ("tn25", tn25_metrics)]:
            s = m["skills"][sk]
            k = s["kinematics"]
            c_all = s["joint_groups"]["calf"]["tau_est"]["ALL"]
            c_po = s["joint_groups"]["calf"]["tau_est"]["Push-off"]
            c_all_pd = s["joint_groups"]["calf"]["tau_pd_raw"]["ALL"]
            row[label] = {
                "stayed_upright": bool(k["stayed_upright"]),
                "safety_ok": bool(s["safety_ok"]),
                "z_start_m": float(k["z_start_m"]),
                "z_max_m": float(k["z_max_m"]),
                "z_end_m": float(k["z_end_m"]),
                "flight_s": float(k["flight"]["flight_s"]),
                "has_flight": bool(k["flight"]["has_flight"]),
                "disp_x_m": float(k["disp_x_m"]),
                "disp_y_m": float(k["disp_y_m"]),
                "flip_angle_deg": k["flip_angle_deg"],
                "end_tilt_deg": float(k["end_tilt_deg"]),
                "max_tilt_deg": float(k["max_tilt_deg"]),
                "calf_max_abs_dq": float(c_all["max_abs_dq"]),
                "calf_max_abs_tau_est": float(c_all["max_abs_tau"]),
                "calf_max_abs_tau_pd": float(c_all_pd["max_abs_tau"]),
                "calf_28p8_viol_4q_pct": float(c_all["28p8V"]["viol_4q_pct"]),
                "calf_28p8_viol_4q_count": int(c_all["28p8V"]["viol_4q_count"]),
                "calf_po_28p8_viol_4q_pct": float(c_po["28p8V"]["viol_4q_pct"]),
                "calf_25p5_viol_4q_pct": float(c_all["25p5V"]["viol_4q_pct"]),
                "calf_25p5_viol_4q_count": int(c_all["25p5V"]["viol_4q_count"]),
                "peak_ibus_est_A": float(s["peak_ibus_est_28p8V_A"]),
            }
        rows[sk] = row

    counts = {
        tag: sum(1 for sk in SKILLS if rows[sk][tag]["stayed_upright"])
        for tag in ("box", "tn28", "tn25")
    }
    return {
        "skills": rows,
        "upright_counts": counts,
        "phys_actuator_params": tn28_metrics["phys_actuator_params"],
    }


def plot_before_after_png(
    box_traces: dict[str, dict[str, Any]],
    tn28_traces: dict[str, dict[str, Any]],
    box_metrics: dict[str, Any],
    tn28_metrics: dict[str, Any],
    summary: dict[str, Any],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.6), dpi=220)
    calf_params = tn28_metrics["phys_actuator_params"]["calf"]
    j_idx = JOINT_GROUPS["calf"]["indices"]
    w_lim, t_lim = 50.0, 62.0

    # Top row: Calf T-omega scatter before (left) vs after 28.8V (right)
    scatter_cfgs = [
        (axes[0, 0], box_traces, box_metrics, "(a) 改造前：仅 XML 矩形限幅 (±45.43 N·m)"),
        (axes[0, 1], tn28_traces, tn28_metrics, "(b) 改造后：启用 28.8 V 四象限 T-ω 包络 (dc_motor)"),
    ]
    for idx, (ax, tr_map, met, title) in enumerate(scatter_cfgs):
        draw_envelopes(ax, calf_params, w_lim, show_legend=(idx == 1), is_calf=True)
        dq_all = np.concatenate([tr_map[s]["dq"][:, j_idx].ravel() for s in SKILLS])
        est_all = np.concatenate([tr_map[s]["tau_est"][:, j_idx].ravel() for s in SKILLS])
        ph_all = np.concatenate([np.repeat(tr_map[s]["phases"], 4) for s in SKILLS])
        for ph in ("Post-stand", "Flight", "Landing", "Push-off"):
            m = ph_all == ph
            if np.any(m):
                ax.scatter(
                    dq_all[m], est_all[m],
                    s=5.0 if ph != "Post-stand" else 3.0,
                    color=PHASE_COLORS[ph],
                    alpha=0.48 if ph != "Post-stand" else 0.16,
                    rasterized=True,
                    label=f"tau_est [{ph}]" if idx == 1 else None,
                    zorder=4,
                )
        agg = met["aggregate"]["calf"]["tau_est"]["ALL"]
        po = met["aggregate"]["calf"]["tau_est"]["Push-off"]
        badge = (
            f"全部阶段 28.8V 越界: {agg['28p8V']['viol_4q_pct']:.2f}% ({agg['28p8V']['viol_4q_count']}点)\n"
            f"Push-off 28.8V 越界: {po['28p8V']['viol_4q_pct']:.2f}%\n"
            f"峰值 |dq|_max: {agg['max_abs_dq']:.2f} rad/s"
        )
        ax.text(
            0.03, 0.96, badge, transform=ax.transAxes, va="top", ha="left", fontsize=8.4,
            bbox={"boxstyle": "round,pad=0.32", "fc": "white", "ec": "#adb5bd", "alpha": 0.93},
            zorder=8,
        )
        ax.set_xlim(-w_lim, w_lim)
        ax.set_ylim(-t_lim, t_lim)
        ax.set_xlabel("膝角速度 dq (rad/s)")
        ax.set_ylabel("实施力矩 τ_est (N·m)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.35)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=5, frameon=True,
        facecolor="white", framealpha=0.95, bbox_to_anchor=(0.5, 0.965), fontsize=8.4,
    )

    # Bottom-left: Apex height z_max across Box / 28.8V / 25.5V
    ax_z = axes[1, 0]
    x = np.arange(len(SKILLS))
    w = 0.26
    z_box = [summary["skills"][sk]["box"]["z_max_m"] for sk in SKILLS]
    z_28 = [summary["skills"][sk]["tn28"]["z_max_m"] for sk in SKILLS]
    z_25 = [summary["skills"][sk]["tn25"]["z_max_m"] for sk in SKILLS]
    b1 = ax_z.bar(x - w, z_box, width=w, color="#6c757d", label="改造前 XML 矩形限幅")
    b2 = ax_z.bar(x, z_28, width=w, color="#1b6ca8", label="改造后 28.8V T-ω")
    b3 = ax_z.bar(x + w, z_25, width=w, color="#e76f51", label="改造后 25.5V 低压 T-ω")
    for bars, tag in [(b1, "box"), (b2, "tn28"), (b3, "tn25")]:
        for bar, sk in zip(bars, SKILLS):
            ok = summary["skills"][sk][tag]["stayed_upright"]
            mark = "稳" if ok else "败"
            col = "#1b4332" if ok else "#9b2226"
            ax_z.text(
                bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.012,
                f"{bar.get_height():.2f}\n{mark}",
                ha="center", va="bottom", fontsize=7.0, color=col, fontweight="bold",
            )
    ax_z.axhline(0.322, color="#495057", ls=":", lw=1.0, label="起跳初始质心 z0=0.322m")
    ax_z.set_xticks(x)
    ax_z.set_xticklabels(SKILLS, rotation=20, ha="right", fontsize=8.5)
    ax_z.set_ylabel("机身最高高度 z_max (m)")
    ax_z.set_ylim(0.0, 0.78)
    ax_z.set_title("(c) 起跳最高高度与落地稳态对比 (稳=站稳, 败=倾倒或塌腰)", fontweight="bold")
    ax_z.grid(True, axis="y", ls=":", alpha=0.35)
    ax_z.legend(loc="upper right", fontsize=8.0, frameon=True)

    # Bottom-right: Flip angle (first 4 skills) + End body tilt (deg) across all 7 skills
    ax_f = axes[1, 1]
    t_box = [summary["skills"][sk]["box"]["end_tilt_deg"] for sk in SKILLS]
    t_28 = [summary["skills"][sk]["tn28"]["end_tilt_deg"] for sk in SKILLS]
    t_25 = [summary["skills"][sk]["tn25"]["end_tilt_deg"] for sk in SKILLS]
    tb1 = ax_f.bar(x - w, t_box, width=w, color="#6c757d", label="改造前 末帧倾角 (°)")
    tb2 = ax_f.bar(x, t_28, width=w, color="#1b6ca8", label="28.8V 末帧倾角 (°)")
    tb3 = ax_f.bar(x + w, t_25, width=w, color="#e76f51", label="25.5V 末帧倾角 (°)")
    ax_f.axhline(30.0, color="#c1121f", ls="--", lw=1.2, label="站稳阈值 30°")
    for idx, sk in enumerate(SKILLS):
        for bar, val in [(tb1[idx], t_box[idx]), (tb2[idx], t_28[idx]), (tb3[idx], t_25[idx])]:
            ax_f.text(
                bar.get_x() + bar.get_width() / 2.0, min(val + 2.0, 172.0),
                f"{val:.0f}°", ha="center", va="bottom", fontsize=6.8,
            )
    ax_f.set_xticks(x)
    ax_f.set_xticklabels(SKILLS, rotation=20, ha="right", fontsize=8.5)
    ax_f.set_ylabel("末帧机身倾角 end_tilt (deg)")
    ax_f.set_ylim(0.0, 185.0)
    ax_f.set_title("(d) 末帧姿态倾角对比 (越小越平稳；30° 以上视为倾倒)", fontweight="bold")
    ax_f.grid(True, axis="y", ls=":", alpha=0.35)
    ax_f.legend(loc="upper right", fontsize=8.0, frameon=True)

    fig.suptitle(
        "unitree_mujoco 加入实物电机 T-ω 曲线前后：7 技能 2 ms Sim2Sim 对比",
        fontsize=12.6, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.905])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def sample_scatter_points(tr: dict[str, dict[str, Any]] | dict[str, Any]) -> list[list[Any]]:
    j_idx = JOINT_GROUPS["calf"]["indices"]
    dq = tr["dq"][:, j_idx].ravel()
    tau_est = tr["tau_est"][:, j_idx].ravel()
    ph = np.repeat(tr["phases"], len(j_idx))

    keep = (ph != "Post-stand") | (np.abs(dq) > 6.0) | (np.abs(tau_est) > 15.0)
    idx_keep = np.flatnonzero(keep)
    idx_bg = np.flatnonzero(~keep)[::8]
    sel = np.sort(np.concatenate([idx_keep, idx_bg]))
    if len(sel) > 3600:
        hi = sel[(np.abs(dq[sel]) > 12.0) | (np.abs(tau_est[sel]) > 20.0)]
        lo = sel[~((np.abs(dq[sel]) > 12.0) | (np.abs(tau_est[sel]) > 20.0))][::2]
        sel = np.sort(np.concatenate([hi, lo]))
    return [
        [round(float(dq[k]), 2), round(float(tau_est[k]), 2), PHASE_ORDER.index(str(ph[k]))]
        for k in sel
    ]


def build_html_payload(
    box_traces: dict[str, dict[str, Any]],
    tn28_traces: dict[str, dict[str, Any]],
    tn25_traces: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    skills_data: dict[str, Any] = {}
    for sk in SKILLS:
        skills_data[sk] = {
            "label": SKILL_LABELS[sk],
            "summary": summary["skills"][sk],
            "points": {
                "box": sample_scatter_points(box_traces[sk]),
                "tn28": sample_scatter_points(tn28_traces[sk]),
                "tn25": sample_scatter_points(tn25_traces[sk]),
            },
        }
    return {
        "skills": list(SKILLS),
        "phase_names": list(PHASE_ORDER),
        "phys_params": summary["phys_actuator_params"]["calf"],
        "upright_counts": summary["upright_counts"],
        "data": skills_data,
    }


def write_comparison_html(payload: dict[str, Any], out_path: Path) -> None:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Go2 实物电机 T-ω 曲线改造前后 2ms Sim2Sim 对比</title>
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
    padding: 4px 8px;
    font-size: 11.5px;
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
    font-size: 14px;
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
    font-size: 12.5px;
    font-weight: 600;
    margin: 0 0 6px 0;
  }}
  canvas {{
    display: block;
    width: 100%;
    height: 250px;
  }}
  canvas#barCanvas {{
    height: 235px;
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
      <div class="btn-row" id="afterBtns">
        <button data-arm="tn28" class="active">28.8V 标称</button>
        <button data-arm="tn25">25.5V 低压</button>
      </div>
    </div>
    <div class="kpi-row" id="kpiRow"></div>
  </div>

  <div class="charts">
    <div class="box">
      <div class="chart-title">改造前：XML 矩形限幅 (膝 Calf τ_est)</div>
      <canvas id="beforeCanvas"></canvas>
    </div>
    <div class="box">
      <div class="chart-title" id="afterTitle">改造后：28.8V T-ω 曲线 (膝 Calf τ_est)</div>
      <canvas id="afterCanvas"></canvas>
    </div>
  </div>

  <div class="box">
    <div class="chart-title">7 技能落地姿态、最高高度 (m) 与翻转角/位移对比（改造前 vs 28.8V vs 25.5V）</div>
    <canvas id="barCanvas"></canvas>
    <div class="legend">
      <span class="leg-item"><span class="dot" style="background:#d95f02"></span>Push-off</span>
      <span class="leg-item"><span class="dot" style="background:#1b6ca8"></span>Flight</span>
      <span class="leg-item"><span class="dot" style="background:#7209b7"></span>Landing</span>
      <span class="leg-item"><span class="dot" style="background:#6c757d"></span>Post-stand</span>
      <span class="leg-item"><span class="line-sw" style="background:#1b6ca8"></span>28.8V 包络</span>
      <span class="leg-item"><span class="line-sw" style="border-top:2px dashed #e76f51"></span>25.5V 包络</span>
      <span class="leg-item"><span class="line-sw" style="background:#2b9348"></span>✓ 站稳 / ✗ 倾倒或过低</span>
    </div>
  </div>
</div>

<script>
const PAYLOAD = {data_json};
let selectedSkill = "ALL";
let afterArm = "tn28"; // "tn28" | "tn25"
const PHASE_COLORS = ["#d95f02", "#1b6ca8", "#7209b7", "#6c757d"];

function cssVar(name, fallback) {{
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}}

function dcBounds(dq, vBus) {{
  const p = PAYLOAD.phys_params;
  const sc = vBus / 28.8;
  const tauSt = p.saturation_effort * sc;
  const w0 = p.velocity_limit * sc;
  const wSat = w0 * (1.0 + p.effort_limit / tauSt);
  const vc = Math.max(-wSat, Math.min(wSat, dq));
  return [
    Math.max(-p.effort_limit, tauSt * (-1.0 - vc / w0)),
    Math.min(p.effort_limit, tauSt * (1.0 - vc / w0)),
  ];
}}

function drawScatter(canvasId, armKey) {{
  const canvas = document.getElementById(canvasId);
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const W = Math.max(240, Math.round(rect.width));
  const H = 250;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const muted = cssVar("--muted-foreground", "#9ca3af");
  const border = cssVar("--border", "#374151");
  const wLim = 48, tLim = 55;
  const pad = {{ l: 36, r: 10, t: 10, b: 26 }};
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const sx = w => pad.l + ((w + wLim) / (2 * wLim)) * pw;
  const sy = t => pad.t + ((tLim - t) / (2 * tLim)) * ph;

  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = border;
  ctx.lineWidth = 0.7;
  ctx.fillStyle = muted;
  ctx.font = "10px monospace";
  for (const wv of [-40, -20, 0, 20, 40]) {{
    const x = sx(wv);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
    ctx.textAlign = "center";
    ctx.fillText(String(wv), x, H - 11);
  }}
  for (const tv of [-45, -20, 0, 20, 45]) {{
    const y = sy(tv);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(String(tv), pad.l - 4, y + 3);
  }}
  ctx.textAlign = "center";
  ctx.fillText("dq (rad/s)", pad.l + pw / 2, H - 1);

  const drawCurve = (vBus, color, dash) => {{
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = dash.length ? 1.5 : 2.0;
    ctx.setLineDash(dash);
    for (const boundIdx of [0, 1]) {{
      ctx.beginPath();
      for (let i = 0; i <= 120; i++) {{
        const w = -wLim + (2 * wLim * i) / 120;
        const b = dcBounds(w, vBus);
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

  const skills = selectedSkill === "ALL" ? PAYLOAD.skills : [selectedSkill];
  for (const sk of skills) {{
    const pts = PAYLOAD.data[sk].points[armKey];
    for (let i = 0; i < pts.length; i++) {{
      const p = pts[i];
      const w = p[0], t = p[1], phId = p[2];
      if (Math.abs(w) > wLim || Math.abs(t) > tLim) continue;
      ctx.fillStyle = PHASE_COLORS[phId];
      ctx.globalAlpha = phId === 3 ? 0.18 : 0.58;
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
  const H = 235;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const fg = cssVar("--foreground", "#e5e7eb");
  const muted = cssVar("--muted-foreground", "#9ca3af");
  const border = cssVar("--border", "#374151");
  const pad = {{ l: 92, r: 196, t: 8, b: 14 }};
  const pw = Math.max(80, W - pad.l - pad.r);
  const ph = H - pad.t - pad.b;
  const n = PAYLOAD.skills.length;
  const rowH = ph / n;
  const maxZ = 0.72;

  ctx.clearRect(0, 0, W, H);
  ctx.font = "11px monospace";

  for (let i = 0; i < n; i++) {{
    const sk = PAYLOAD.skills[i];
    const row = PAYLOAD.data[sk].summary;
    const y = pad.t + i * rowH;

    ctx.fillStyle = sk === selectedSkill ? cssVar("--primary", "#3b82f6") : fg;
    ctx.textAlign = "right";
    ctx.fillText(sk, pad.l - 8, y + rowH * 0.62);

    ctx.fillStyle = border;
    ctx.fillRect(pad.l, y + rowH * 0.16, pw, rowH * 0.68);

    const arms = [
      ["box", "#6c757d", 0.18],
      ["tn28", "#1b6ca8", 0.42],
      ["tn25", "#e76f51", 0.66],
    ];
    for (const [k, col, frac] of arms) {{
      const s = row[k];
      const bw = Math.min(pw, (s.z_max_m / maxZ) * pw);
      ctx.fillStyle = col;
      ctx.fillRect(pad.l, y + rowH * frac, Math.max(2, bw), rowH * 0.18);
    }}

    const b = row.box, a = row[afterArm];
    const bMk = b.stayed_upright ? "✓" : "✗";
    const aMk = a.stayed_upright ? "✓" : "✗";
    const extra = a.flip_angle_deg !== null
      ? `${{b.flip_angle_deg.toFixed(0)}}→${{a.flip_angle_deg !== null ? a.flip_angle_deg.toFixed(0) + "°" : "—"}}`
      : `x=${{a.disp_x_m >= 0 ? "+" : ""}}${{a.disp_x_m.toFixed(2)}}m`;
    ctx.fillStyle = a.stayed_upright ? "#2b9348" : (b.stayed_upright ? "#e76f51" : muted);
    ctx.textAlign = "left";
    ctx.fillText(
      `${{b.z_max_m.toFixed(2)}}→${{a.z_max_m.toFixed(2)}}m ${{bMk}}→${{aMk}} ${{extra}}`,
      pad.l + pw + 6, y + rowH * 0.62
    );
  }}
}}

function updateKPIs() {{
  const skills = selectedSkill === "ALL" ? PAYLOAD.skills : [selectedSkill];
  const armTitle = afterArm === "tn28" ? "28.8V" : "25.5V";
  document.getElementById("afterTitle").textContent = `改造后：${{armTitle}} T-ω 曲线 (膝 Calf τ_est)`;

  let boxOk = 0, afterOk = 0, violBefore = 0, violAfter = 0;
  let dqBefore = 0, dqAfter = 0;
  for (const sk of skills) {{
    const b = PAYLOAD.data[sk].summary.box;
    const a = PAYLOAD.data[sk].summary[afterArm];
    if (b.stayed_upright) boxOk++;
    if (a.stayed_upright) afterOk++;
    violBefore = Math.max(violBefore, afterArm === "tn28" ? b.calf_28p8_viol_4q_pct : b.calf_25p5_viol_4q_pct);
    violAfter = Math.max(violAfter, afterArm === "tn28" ? a.calf_28p8_viol_4q_pct : a.calf_25p5_viol_4q_pct);
    dqBefore = Math.max(dqBefore, b.calf_max_abs_dq);
    dqAfter = Math.max(dqAfter, a.calf_max_abs_dq);
  }}

  const okTxt = selectedSkill === "ALL"
    ? `${{boxOk}}/${{skills.length}} → ${{afterOk}}/${{skills.length}}`
    : `${{boxOk ? "站稳" : "倾倒"}} → ${{afterOk ? "站稳" : "倾倒"}}`;
  const items = [
    ["站稳技能数 (前→后)", okTxt],
    [`${{armTitle}} 膝越界峰值`, `${{violBefore.toFixed(2)}}% → ${{violAfter.toFixed(2)}}%`],
    ["膝峰值 |dq| (前→后)", `${{dqBefore.toFixed(1)}} → ${{dqAfter.toFixed(1)}} rad/s`],
    ["当前对照分支", `${{armTitle}} dc_motor`],
  ];
  document.getElementById("kpiRow").innerHTML = items.map(([l, v]) =>
    `<div class="kpi"><div class="lbl">${{l}}</div><div class="val">${{v}}</div></div>`
  ).join("");
}}

function renderAll() {{
  updateKPIs();
  drawScatter("beforeCanvas", "box");
  drawScatter("afterCanvas", afterArm);
  drawBars();
}}

function init() {{
  const sRow = document.getElementById("skillBtns");
  const allBtn = document.createElement("button");
  allBtn.textContent = "全部";
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

  const aRow = document.getElementById("afterBtns");
  aRow.querySelectorAll("button").forEach(btn => {{
    btn.onclick = () => {{
      afterArm = btn.dataset.arm;
      aRow.querySelectorAll("button").forEach(b => b.classList.remove("active"));
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


def write_comparison_markdown(summary: dict[str, Any], out_dir: Path, box_dir: Path, tn28_dir: Path, tn25_dir: Path) -> None:
    lines = [
        "# Go2 `unitree_mujoco` 实物电机 T-ω 曲线改造前后 2 ms Sim2Sim 对比",
        "",
        f"- 改造前（仅 XML 矩形限幅）: `{box_dir}`",
        f"- 改造后 `28.8 V` 标称 (`dc_motor`): `{tn28_dir}`",
        f"- 改造后 `25.5 V` 低压 (`dc_motor`): `{tn25_dir}`",
        "",
        "## 1. 7 技能落地结果对比",
        "",
        "| 技能 | 改造前 (XML 矩形) | 改造后 `28.8 V` `dc_motor` | 改造后 `25.5 V` 低压 | 最高高度 `z_max` (前 / 28.8V / 25.5V) | 末帧高度 `z_end` (前 / 28.8V / 25.5V) | 末帧倾角 (前 / 28.8V / 25.5V) | 翻转角或 X 位移 (前 / 28.8V / 25.5V) | 膝 28.8V 越界 (前 → 后) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for sk in SKILLS:
        row = summary["skills"][sk]
        b, n, l = row["box"], row["tn28"], row["tn25"]
        fmt_ok = lambda r: "✅ 站稳" if r["stayed_upright"] else "❌ 失败"
        if b["flip_angle_deg"] is not None or n["flip_angle_deg"] is not None:
            f_str = lambda r: f"{r['flip_angle_deg']:.1f}°" if r["flip_angle_deg"] is not None else "—"
            motion = f"{f_str(b)} / {f_str(n)} / {f_str(l)}"
        else:
            motion = f"{b['disp_x_m']:+.2f}m / {n['disp_x_m']:+.2f}m / {l['disp_x_m']:+.2f}m"
        lines.append(
            f"| `{sk}` | {fmt_ok(b)} | {fmt_ok(n)} | {fmt_ok(l)} | "
            f"{b['z_max_m']:.3f} / {n['z_max_m']:.3f} / {l['z_max_m']:.3f} | "
            f"{b['z_end_m']:.3f} / {n['z_end_m']:.3f} / {l['z_end_m']:.3f} | "
            f"{b['end_tilt_deg']:.1f}° / {n['end_tilt_deg']:.1f}° / {l['end_tilt_deg']:.1f}° | "
            f"{motion} | "
            f"{b['calf_28p8_viol_4q_pct']:.2f}% → {n['calf_28p8_viol_4q_pct']:.2f}% |"
        )
    lines.extend([
        "",
        "## 2. 诊断产物",
        "",
        "- [comparison_before_after.json](./comparison_before_after.json)",
        "- [fig3_before_after_comparison.png](./fig3_before_after_comparison.png)",
        "- [comparison_interactive.html](./comparison_interactive.html)",
        "",
    ])
    (out_dir / "comparison_README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--box-dir", type=Path,
        default=Path("/home/rkz/code/unitree_ws/results/My_unitree_go2_gym_flip_mjlab_20261123"),
    )
    parser.add_argument(
        "--tn28-dir", type=Path,
        default=Path("/home/rkz/code/unitree_ws/results/My_unitree_go2_gym_flip_mjlab_tn28p8_20261124"),
    )
    parser.add_argument(
        "--tn25-dir", type=Path,
        default=Path("/home/rkz/code/unitree_ws/results/My_unitree_go2_gym_flip_mjlab_tn25p5_20261124"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or args.tn28_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_cjk_matplotlib()
    box_m, box_t = load_run_bundle(args.box_dir)
    tn28_m, tn28_t = load_run_bundle(args.tn28_dir)
    tn25_m, tn25_t = load_run_bundle(args.tn25_dir)

    summary = build_comparison_summary(box_m, tn28_m, tn25_m)
    (out_dir / "comparison_before_after.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_before_after_png(box_t, tn28_t, box_m, tn28_m, summary, out_dir / "fig3_before_after_comparison.png")
    payload = build_html_payload(box_t, tn28_t, tn25_t, summary)
    write_comparison_html(payload, out_dir / "comparison_interactive.html")
    write_comparison_markdown(summary, out_dir, args.box_dir, args.tn28_dir, args.tn25_dir)
    print(f"[OK] Wrote comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
