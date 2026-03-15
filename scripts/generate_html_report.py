"""Generate an interactive HTML evaluation report.

Creates a self-contained HTML file with:
- Summary metrics (FID, LPIPS, NME, SSIM, ArcFace)
- Per-procedure breakdown tables
- Per-Fitzpatrick type fairness analysis
- Embedded comparison images (base64-encoded)

Usage:
    python scripts/generate_html_report.py \
        --eval_results eval_results.json \
        --baselines results/baselines.json \
        --images_dir results/comparison_images/ \
        --output results/report.html
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2


def img_to_base64(path: str, max_width: int = 400) -> str:
    """Read image and convert to base64 data URI."""
    img = cv2.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def metric_cell(val: float | None, fmt: str = ".4f") -> str:
    """Format a metric value with color coding."""
    if val is None or val == 0.0:
        return '<td class="na">--</td>'
    formatted = f"{val:{fmt}}"
    return f"<td>{formatted}</td>"


def generate_html(
    eval_results: dict | None = None,
    baselines: dict | None = None,
    images_dir: str | None = None,
    title: str = "LandmarkDiff Evaluation Report",
) -> str:
    """Generate HTML report string."""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
    h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
    h2 {{ color: #16213e; margin-top: 30px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    th {{ background: #16213e; color: white; padding: 10px 15px; text-align: left; }}
    td {{ padding: 8px 15px; border-bottom: 1px solid #eee; }}
    tr:hover {{ background: #f8f9fa; }}
    .na {{ color: #999; }}
    .summary-card {{ display: inline-block; background: white; padding: 20px;
                     margin: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                     min-width: 150px; text-align: center; }}
    .summary-card .value {{ font-size: 28px; font-weight: bold; color: #16213e; }}
    .summary-card .label {{ font-size: 14px; color: #666; margin-top: 5px; }}
    .comparison-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                        gap: 15px; margin: 20px 0; }}
    .comparison-item {{ background: white; padding: 10px; border-radius: 8px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .comparison-item img {{ width: 100%; border-radius: 4px; }}
    .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd;
               color: #999; font-size: 12px; }}
</style>
</head>
<body>
<h1>{title}</h1>
"""

    # Summary cards
    metrics = {}
    if eval_results:
        metrics = eval_results.get("metrics", {})
    elif baselines:
        for method in baselines.get("methods", {}).values():
            metrics = method.get("metrics", {})
            break

    if metrics:
        html += '<div style="text-align: center; margin: 20px 0;">\n'
        for name, key, fmt_str in [
            ("FID", "fid", ".1f"),
            ("SSIM", "ssim", ".4f"),
            ("LPIPS", "lpips", ".4f"),
            ("NME", "nme", ".4f"),
            ("Identity Sim", "identity_sim", ".4f"),
        ]:
            val = metrics.get(key)
            if val is not None and val != 0.0:
                html += f'<div class="summary-card"><div class="value">{val:{fmt_str}}</div>'
                html += f'<div class="label">{name}</div></div>\n'
        html += "</div>\n"

    # Main results table
    html += "<h2>Quantitative Results</h2>\n"
    html += "<table>\n<tr><th>Method</th><th>FID</th><th>SSIM</th>"
    html += "<th>LPIPS</th><th>NME</th><th>ArcFace</th></tr>\n"

    if baselines:
        for method_key, method_name in [("tps", "TPS-only"), ("morphing", "Morphing")]:
            m = baselines.get("methods", {}).get(method_key, {}).get("metrics", {})
            html += f"<tr><td>{method_name}</td>"
            html += metric_cell(m.get("fid"), ".1f")
            html += metric_cell(m.get("ssim"))
            html += metric_cell(m.get("lpips"))
            html += metric_cell(m.get("nme"))
            html += metric_cell(m.get("identity_sim"))
            html += "</tr>\n"

    if eval_results:
        m = eval_results.get("metrics", {})
        html += '<tr style="background:#e8f5e9; font-weight:bold;"><td>LandmarkDiff (Ours)</td>'
        html += metric_cell(m.get("fid"), ".1f")
        html += metric_cell(m.get("ssim"))
        html += metric_cell(m.get("lpips"))
        html += metric_cell(m.get("nme"))
        html += metric_cell(m.get("identity_sim"))
        html += "</tr>\n"

    html += "</table>\n"

    # Per-procedure breakdown
    source = eval_results or {}
    per_proc = source.get("per_procedure", {})
    if not per_proc and baselines:
        for method in baselines.get("methods", {}).values():
            per_proc = method.get("per_procedure", {})
            break

    if per_proc:
        html += "<h2>Per-Procedure Breakdown</h2>\n"
        html += "<table>\n<tr><th>Procedure</th><th>SSIM</th>"
        html += "<th>LPIPS</th><th>NME</th>"
        html += "<th>ArcFace</th><th>N</th></tr>\n"
        for proc in ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]:
            vals = per_proc.get(proc, {})
            if not vals:
                continue
            html += f"<tr><td>{proc.capitalize()}</td>"
            html += metric_cell(vals.get("ssim"))
            html += metric_cell(vals.get("lpips"))
            html += metric_cell(vals.get("nme"))
            html += metric_cell(vals.get("identity_sim"))
            html += f"<td>{vals.get('count', 0)}</td></tr>\n"
        html += "</table>\n"

    # Fitzpatrick fairness table
    per_fitz = source.get("per_fitzpatrick", {})
    if not per_fitz and baselines:
        for method in baselines.get("methods", {}).values():
            per_fitz = method.get("per_fitzpatrick", {})
            break

    if per_fitz:
        html += "<h2>Fairness Analysis (Fitzpatrick Skin Type)</h2>\n"
        html += "<table>\n<tr><th>Type</th><th>SSIM</th>"
        html += "<th>LPIPS</th><th>NME</th>"
        html += "<th>ArcFace</th><th>N</th></tr>\n"
        for ft in ["I", "II", "III", "IV", "V", "VI"]:
            vals = per_fitz.get(ft, {})
            if not vals:
                continue
            html += f"<tr><td>Type {ft}</td>"
            html += metric_cell(vals.get("ssim"))
            html += metric_cell(vals.get("lpips"))
            html += metric_cell(vals.get("nme"))
            html += metric_cell(vals.get("identity_sim"))
            html += f"<td>{vals.get('count', 0)}</td></tr>\n"
        html += "</table>\n"

    # Comparison images
    if images_dir and Path(images_dir).exists():
        images = sorted(Path(images_dir).glob("*.png"))[:20]
        if images:
            html += "<h2>Sample Comparisons</h2>\n"
            html += '<div class="comparison-grid">\n'
            for img_path in images:
                b64 = img_to_base64(str(img_path))
                if b64:
                    html += '<div class="comparison-item">'
                    html += f'<img src="{b64}" alt="{img_path.stem}">'
                    html += '<div style="text-align:center;font-size:12px;color:#666;">'
                    html += f"{img_path.stem}</div></div>\n"
            html += "</div>\n"

    num_pairs = eval_results.get("num_pairs", 0) if eval_results else 0
    html += f"""
<div class="footer">
Generated by LandmarkDiff evaluation pipeline.<br>
Report includes {num_pairs} evaluation pairs.
</div>
</body>
</html>"""

    return html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML evaluation report")
    parser.add_argument("--eval_results", default=None)
    parser.add_argument("--baselines", default=None)
    parser.add_argument("--images_dir", default=None)
    parser.add_argument("--output", default="results/report.html")
    parser.add_argument("--title", default="LandmarkDiff Evaluation Report")
    args = parser.parse_args()

    eval_results = None
    baselines = None
    if args.eval_results and Path(args.eval_results).exists():
        with open(args.eval_results) as f:
            eval_results = json.load(f)
    if args.baselines and Path(args.baselines).exists():
        with open(args.baselines) as f:
            baselines = json.load(f)

    html = generate_html(eval_results, baselines, args.images_dir, args.title)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Report saved to {args.output}")
