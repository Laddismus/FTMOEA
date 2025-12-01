from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from afts_pro.core.benchmark_report import BenchmarkComparison, BenchmarkReport


@dataclass
class HtmlReportConfig:
    title: str = "AFTS Benchmark Report"
    include_equity_chart: bool = True
    include_drawdown_chart: bool = True
    include_daily_pnl_chart: bool = True
    include_ftmo_table: bool = True
    include_rl_train_section: bool = True
    equity_csv_name: str = "equity.csv"
    daily_pnl_csv_name: str = "daily_pnl.csv"


class BenchmarkHtmlRenderer:
    def __init__(self, cfg: HtmlReportConfig) -> None:
        self.cfg = cfg

    def render_single(self, report: BenchmarkReport, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        eq_points = self._load_series(run_dir / self.cfg.equity_csv_name)
        dd_points: List[Tuple[float, float]] = []
        if eq_points:
            dd_points = self._compute_drawdown(eq_points)
        daily_points = self._load_series(run_dir / self.cfg.daily_pnl_csv_name)

        html_parts = [
            "<html><head>",
            f"<title>{self.cfg.title}</title>",
            "<style>body{font-family:Arial, sans-serif;margin:20px;} .section{margin-bottom:24px;} table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:4px 8px;} .best{font-weight:bold;color:#0a0;}</style>",
            "</head><body>",
            f"<h1>{self.cfg.title}</h1>",
            f"<p>Checkpoint: {report.checkpoint_path} | Score: {report.score:.4f}</p>",
            self._render_kpi_table(report),
        ]
        if self.cfg.include_ftmo_table:
            html_parts.append(self._render_ftmo_table(report))
        if self.cfg.include_rl_train_section:
            html_parts.append(self._render_rl_train(report))
        if self.cfg.include_equity_chart and eq_points:
            html_parts.append(self._render_chart(eq_points, "Equity Curve", color="#007acc"))
        if self.cfg.include_drawdown_chart and dd_points:
            html_parts.append(self._render_chart(dd_points, "Drawdown Curve", color="#cc0000"))
        if self.cfg.include_daily_pnl_chart and daily_points:
            html_parts.append(self._render_chart(daily_points, "Daily PnL", color="#ff9900"))
        html_parts.append("</body></html>")
        html_str = "\n".join(html_parts)
        out_path = run_dir / "benchmark_report.html"
        out_path.write_text(html_str, encoding="utf-8")
        return out_path

    def render_comparison(self, comparison: BenchmarkComparison, out_path: Path) -> Path:
        rows = []
        for rep in comparison.ranked:
            label = "BEST" if rep.checkpoint_path == comparison.best_checkpoint else ""
            rows.append(
                f"<tr><td>{rep.checkpoint_path}</td><td>{rep.score:.4f}</td><td>{rep.kpis.get('profit_factor', 0.0):.3f}</td><td>{rep.kpis.get('winrate', 0.0):.3f}</td><td>{rep.kpis.get('mdd_pct', 0.0):.3f}</td><td>{rep.ftmo.get('ftmo_pass', False)}</td><td>{label}</td></tr>"
            )
        html = "\n".join(
            [
                "<html><head><title>Benchmark Comparison</title></head><body>",
                "<h1>Benchmark Comparison</h1>",
                "<table>",
                "<tr><th>Checkpoint</th><th>Score</th><th>PF</th><th>Winrate</th><th>MDD%</th><th>FTMO Pass</th><th></th></tr>",
                *rows,
                "</table>",
                "</body></html>",
            ]
        )
        out_path.write_text(html, encoding="utf-8")
        return out_path

    def _render_kpi_table(self, report: BenchmarkReport) -> str:
        k = report.kpis
        return "\n".join(
            [
                '<div class="section">',
                "<h2>KPIs</h2>",
                f'<div style="display:none">keys:{",".join(k.keys())}</div>',
                "<table>",
                "<tr><th>Metric</th><th>Value</th></tr>",
                f"<tr><td>Profit Factor</td><td>{k.get('profit_factor', 0):.3f}</td></tr>",
                f"<tr><td>Winrate</td><td>{k.get('winrate', 0):.3f}</td></tr>",
                f"<tr><td>Max Drawdown %</td><td>{k.get('mdd_pct', 0):.3f}</td></tr>",
                f"<tr><td>MAR</td><td>{k.get('mar', 0):.3f}</td></tr>",
                f"<tr><td>Sharpe-like</td><td>{k.get('sharpe_like', 0):.3f}</td></tr>",
                f"<tr><td>Trades</td><td>{k.get('trades', 0)}</td></tr>",
                "</table>",
                "</div>",
            ]
        )

    def _render_ftmo_table(self, report: BenchmarkReport) -> str:
        f = report.ftmo
        return "\n".join(
            [
                '<div class="section">',
                "<h2>FTMO Metrics</h2>",
                "<table>",
                "<tr><th>Metric</th><th>Value</th></tr>",
                f"<tr><td>FTMO Pass</td><td>{f.get('ftmo_pass', False)}</td></tr>",
                f"<tr><td>Daily DD Max</td><td>{f.get('daily_dd', f.get('daily_dd_max', 0))}</td></tr>",
                f"<tr><td>Overall DD Max</td><td>{f.get('overall_dd', f.get('overall_dd_max', 0))}</td></tr>",
                f"<tr><td>Target Progress %</td><td>{f.get('target_progress_pct', 0)}</td></tr>",
                "</table>",
                "</div>",
            ]
        )

    def _render_rl_train(self, report: BenchmarkReport) -> str:
        r = report.rl_train
        return "\n".join(
            [
                '<div class="section">',
                "<h2>RL Training</h2>",
                "<table>",
                "<tr><th>Metric</th><th>Value</th></tr>",
                f"<tr><td>Mean Reward</td><td>{r.get('mean_reward', 0):.3f}</td></tr>",
                f"<tr><td>Best Reward</td><td>{r.get('best_reward', 0):.3f}</td></tr>",
                f"<tr><td>Reward Slope</td><td>{r.get('reward_slope', 0):.3f}</td></tr>",
                f"<tr><td>Epsilon Mean</td><td>{r.get('epsilon_mean', 0):.3f}</td></tr>",
                "</table>",
                "</div>",
            ]
        )

    def _load_series(self, path: Path) -> List[Tuple[float, float]]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text().splitlines():
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                rows.append((x, y))
            except ValueError:
                continue
        return rows

    def _compute_drawdown(self, series: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
        dd = []
        max_eq = None
        for x, y in series:
            max_eq = y if max_eq is None else max(max_eq, y)
            dd.append((x, max(0.0, (max_eq - y))))
        return dd

    def _render_chart(self, pts: List[Tuple[float, float]], title: str, color: str = "#007acc") -> str:
        if not pts:
            return ""
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_y = min(ys)
        max_y = max(ys) if ys else 1.0
        span_y = max(max_y - min_y, 1e-6)
        width = 600
        height = 200
        svg_points = []
        for i, (x, y) in enumerate(pts):
            px = (i / max(len(pts) - 1, 1)) * width
            py = height - ((y - min_y) / span_y) * height
            svg_points.append(f"{px:.2f},{py:.2f}")
        poly = " ".join(svg_points)
        return "\n".join(
            [
                '<div class="section">',
                f"<h3>{title}</h3>",
                f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
                f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{poly}" />',
                "</svg>",
                "</div>",
            ]
        )
