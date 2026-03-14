#!/usr/bin/env python3
"""Self-contained HTML training dashboard — works without WandB.

Reads SLURM training logs and generates an interactive HTML page with:
1. Loss curve (with smoothing slider)
2. Learning rate schedule
3. Gradient norm history
4. Training speed (it/s, ETA)
5. Checkpoint timeline
6. Convergence analysis
7. Auto-refresh mode for live monitoring

Usage:
    # Generate dashboard from SLURM log
    python scripts/training_dashboard.py --log slurm-phaseA-12345.out

    # Auto-detect latest log
    python scripts/training_dashboard.py

    # Custom output
    python scripts/training_dashboard.py --log slurm-*.out --output dashboard.html

    # Watch mode (regenerate every N seconds)
    python scripts/training_dashboard.py --watch 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def find_latest_log() -> str | None:
    """Find the most recent SLURM training log."""
    candidates = sorted(
        PROJECT_ROOT.glob("slurm-*.out"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    return None


def parse_log_data(log_path: str) -> dict:
    """Parse training log into structured data for the dashboard."""
    from scripts.monitor_training import detect_convergence, parse_training_log

    data = parse_training_log(log_path)
    convergence = (
        detect_convergence(data) if len(data.get("steps", [])) >= 5 else "Too few data points"
    )

    # Extract header info
    header = {}
    try:
        with open(log_path) as f:
            for line in f:
                if "Phase" in line and "Training" in line:
                    header["phase"] = line.strip()
                elif "GPU:" in line:
                    header["gpu"] = line.strip().split("GPU:", 1)[-1].strip()
                elif "Dataset:" in line:
                    header["dataset"] = line.strip().split("Dataset:", 1)[-1].strip()
                elif "Device:" in line:
                    header["device"] = line.strip()
                elif "Optimizer steps:" in line:
                    header["optimizer"] = line.strip()
                if len(header) >= 5:
                    break
    except Exception:
        pass

    return {
        "steps": data.get("steps", []),
        "losses": data.get("losses", []),
        "learning_rates": data.get("learning_rates", []),
        "grad_norms": data.get("grad_norms", []),
        "speeds": data.get("speeds", []),
        "etas": data.get("etas", []),
        "total_steps": data.get("total_steps", 0),
        "checkpoints": data.get("checkpoints", []),
        "convergence": convergence,
        "header": header,
        "log_path": log_path,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_dashboard_html(data: dict) -> str:
    """Generate a self-contained HTML dashboard."""
    steps_json = json.dumps(data["steps"])
    losses_json = json.dumps(data["losses"])
    lrs_json = json.dumps(data["learning_rates"])
    grads_json = json.dumps(data["grad_norms"])
    speeds_json = json.dumps(data["speeds"])
    total_steps = data["total_steps"]
    n_steps = len(data["steps"])

    # Summary stats
    current_step = data["steps"][-1] if data["steps"] else 0
    current_loss = data["losses"][-1] if data["losses"] else 0
    min_loss = min(data["losses"]) if data["losses"] else 0
    current_lr = data["learning_rates"][-1] if data["learning_rates"] else 0
    avg_speed = sum(data["speeds"]) / len(data["speeds"]) if data["speeds"] else 0
    progress_pct = (current_step / total_steps * 100) if total_steps > 0 else 0
    last_eta = data["etas"][-1] if data["etas"] else "?"

    # Checkpoint info
    ckpt_html = ""
    for ckpt in data["checkpoints"]:
        ckpt_html += f'<div class="ckpt-badge">{Path(ckpt).name}</div>\n'

    header = data.get("header", {})
    phase_info = header.get("phase", "Unknown Phase")
    gpu_info = header.get("gpu", "Unknown GPU")
    dataset_info = header.get("dataset", "Unknown dataset")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LandmarkDiff Training Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
.header {{ text-align: center; padding: 20px 0; border-bottom: 1px solid #21262d; margin-bottom: 20px; }}
.header h1 {{ color: #58a6ff; font-size: 24px; }}
.header .meta {{ color: #8b949e; font-size: 13px; margin-top: 8px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }}
.stat {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px; text-align: center; }}
.stat .value {{ font-size: 28px; font-weight: 700; color: #58a6ff; }}
.stat .label {{ font-size: 12px; color: #8b949e; margin-top: 4px; text-transform: uppercase; }}
.progress-bar {{ background: #21262d; border-radius: 8px; height: 24px; margin-bottom: 20px; overflow: hidden; position: relative; }}
.progress-fill {{ background: linear-gradient(90deg, #238636, #58a6ff); height: 100%; border-radius: 8px; transition: width 0.3s; }}
.progress-text {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 12px; font-weight: 600; }}
.chart-container {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
.chart-container h3 {{ color: #e6edf3; margin-bottom: 12px; font-size: 15px; }}
canvas {{ width: 100% !important; height: 250px !important; }}
.controls {{ display: flex; gap: 12px; align-items: center; margin-bottom: 8px; }}
.controls label {{ font-size: 12px; color: #8b949e; }}
.controls input[type="range"] {{ flex: 1; }}
.convergence {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
.convergence h3 {{ margin-bottom: 8px; }}
.convergence .analysis {{ font-family: monospace; white-space: pre-wrap; color: #8b949e; font-size: 13px; }}
.ckpt-section {{ margin-bottom: 16px; }}
.ckpt-badge {{ display: inline-block; background: #238636; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; margin: 4px; }}
.footer {{ text-align: center; color: #484f58; font-size: 11px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #21262d; }}
</style>
</head>
<body>
<div class="header">
    <h1>LandmarkDiff Training Dashboard</h1>
    <div class="meta">
        {phase_info} | {gpu_info} | {dataset_info}<br>
        Log: {Path(data["log_path"]).name} | Generated: {data["generated"]}
    </div>
</div>

<div class="progress-bar">
    <div class="progress-fill" style="width: {progress_pct:.1f}%"></div>
    <div class="progress-text">{current_step:,} / {total_steps:,} steps ({progress_pct:.1f}%)</div>
</div>

<div class="grid">
    <div class="stat"><div class="value">{current_step:,}</div><div class="label">Current Step</div></div>
    <div class="stat"><div class="value">{current_loss:.4f}</div><div class="label">Current Loss</div></div>
    <div class="stat"><div class="value">{min_loss:.4f}</div><div class="label">Min Loss</div></div>
    <div class="stat"><div class="value">{current_lr:.2e}</div><div class="label">Learning Rate</div></div>
    <div class="stat"><div class="value">{avg_speed:.1f}</div><div class="label">Avg it/s</div></div>
    <div class="stat"><div class="value">{last_eta}</div><div class="label">ETA</div></div>
</div>

<div class="chart-container">
    <h3>Training Loss</h3>
    <div class="controls">
        <label>Smoothing: <span id="smooth-val">0.9</span></label>
        <input type="range" id="smooth-slider" min="0" max="0.99" step="0.01" value="0.9"
               oninput="document.getElementById('smooth-val').textContent=this.value; drawLoss()">
    </div>
    <canvas id="lossChart"></canvas>
</div>

<div class="chart-container">
    <h3>Learning Rate Schedule</h3>
    <canvas id="lrChart"></canvas>
</div>

<div class="chart-container">
    <h3>Gradient Norm</h3>
    <canvas id="gradChart"></canvas>
</div>

<div class="chart-container">
    <h3>Training Speed (it/s)</h3>
    <canvas id="speedChart"></canvas>
</div>

<div class="convergence">
    <h3>Convergence Analysis</h3>
    <div class="analysis">{data["convergence"]}</div>
</div>

{"<div class='ckpt-section'><h3>Checkpoints</h3>" + ckpt_html + "</div>" if ckpt_html else ""}

<div class="footer">
    LandmarkDiff Training Dashboard &mdash; {n_steps} data points &mdash; Auto-generated
</div>

<script>
const steps = {steps_json};
const losses = {losses_json};
const lrs = {lrs_json};
const grads = {grads_json};
const speeds = {speeds_json};

function ema(data, alpha) {{
    if (!data.length) return [];
    let result = [data[0]];
    for (let i = 1; i < data.length; i++) {{
        result.push(alpha * result[i-1] + (1 - alpha) * data[i]);
    }}
    return result;
}}

function drawChart(canvasId, xs, ys, color, label, logScale) {{
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    const pad = {{top: 10, right: 20, bottom: 30, left: 60}};
    const pW = W - pad.left - pad.right;
    const pH = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    if (!xs.length || !ys.length) {{
        ctx.fillStyle = '#484f58';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No data', W/2, H/2);
        return;
    }}

    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    let yMin = Math.min(...ys), yMax = Math.max(...ys);
    if (logScale && yMin > 0) {{
        yMin = Math.log10(yMin);
        yMax = Math.log10(yMax);
    }}
    const yRange = yMax - yMin || 1;
    yMin -= yRange * 0.05;
    yMax += yRange * 0.05;

    function xPos(v) {{ return pad.left + (v - xMin) / ((xMax - xMin) || 1) * pW; }}
    function yPos(v) {{
        let val = logScale && v > 0 ? Math.log10(v) : v;
        return pad.top + pH - (val - yMin) / (yMax - yMin) * pH;
    }}

    // Grid
    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {{
        let y = pad.top + pH * i / 4;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
        let val = yMax - (yMax - yMin) * i / 4;
        if (logScale) val = Math.pow(10, val);
        ctx.fillStyle = '#484f58';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(val.toExponential(2), pad.left - 5, y + 3);
    }}

    // X-axis labels
    ctx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) {{
        let x = pad.left + pW * i / 4;
        let val = xMin + (xMax - xMin) * i / 4;
        ctx.fillText(Math.round(val).toLocaleString(), x, H - 5);
    }}

    // Line
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {{
        let x = xPos(xs[i]), y = yPos(ys[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }}
    ctx.stroke();
}}

function drawLoss() {{
    const alpha = parseFloat(document.getElementById('smooth-slider').value);
    const smoothed = ema(losses, alpha);
    drawChart('lossChart', steps, smoothed, '#f0883e', 'Loss', false);
}}

drawLoss();
drawChart('lrChart', steps, lrs, '#58a6ff', 'LR', true);
drawChart('gradChart', steps, grads, '#bc8cff', 'GradNorm', false);
drawChart('speedChart', steps, speeds, '#3fb950', 'it/s', false);

window.addEventListener('resize', () => {{
    drawLoss();
    drawChart('lrChart', steps, lrs, '#58a6ff', 'LR', true);
    drawChart('gradChart', steps, grads, '#bc8cff', 'GradNorm', false);
    drawChart('speedChart', steps, speeds, '#3fb950', 'it/s', false);
}});
</script>
</body>
</html>"""

    return html


def generate_dashboard(
    log_path: str | None = None,
    output_path: str | None = None,
) -> str:
    """Generate an HTML training dashboard from a SLURM log."""
    if log_path is None:
        log_path = find_latest_log()
        if log_path is None:
            logger.error("No SLURM training logs found")
            return ""

    if not Path(log_path).exists():
        logger.error("Log not found: %s", log_path)
        return ""

    logger.info("Parsing: %s", Path(log_path).name)
    data = parse_log_data(log_path)
    logger.info(
        "Found %d data points, %d checkpoints", len(data["steps"]), len(data["checkpoints"])
    )

    html = generate_dashboard_html(data)

    if output_path is None:
        stem = Path(log_path).stem
        output_path = str(PROJECT_ROOT / f"dashboard_{stem}.html")

    Path(output_path).write_text(html)
    logger.info("Dashboard: %s", output_path)

    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Training dashboard generator")
    parser.add_argument("--log", default=None, help="SLURM log file (auto-detect if omitted)")
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument(
        "--watch", type=int, default=0, help="Watch mode: regenerate every N seconds"
    )
    args = parser.parse_args()

    if args.watch > 0:
        logger.info("Watch mode: regenerating every %ds (Ctrl+C to stop)", args.watch)
        while True:
            try:
                generate_dashboard(args.log, args.output)
                time.sleep(args.watch)
            except KeyboardInterrupt:
                logger.info("Stopped.")
                break
    else:
        path = generate_dashboard(args.log, args.output)
        if not path:
            sys.exit(1)
