from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from afts_pro.core.benchmark_engine import BenchmarkEngine
from afts_pro.core.benchmark_report import BenchmarkComparison, BenchmarkReport
from afts_pro.core.benchmark_html import BenchmarkHtmlRenderer, HtmlReportConfig

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    base: Dict[str, str]
    rl: Dict[str, str | bool]
    output: Dict[str, str | bool]

    @staticmethod
    def from_yaml(path: str) -> "EvalConfig":
        data = yaml.safe_load(Path(path).read_text())
        return EvalConfig(
            base=data.get("base", {}),
        rl=data.get("rl", {}),
        output=data.get("output", {}),
        )


class EvalController:
    """
    Orchestrates evaluation runs for checkpoints.
    """

    def __init__(self, eval_config: EvalConfig):
        self.cfg = eval_config
        self.engine = BenchmarkEngine()

    def _default_metrics(self) -> Dict[str, float]:
        return {
            "profit_factor": 1.0,
            "winrate": 0.5,
            "mdd_pct": 5.0,
            "mar": 0.2,
        }

    def _default_ftmo_metrics(self) -> Dict[str, float | bool]:
        return {"ftmo_pass": True, "daily_dd_max": 3.0, "overall_dd_max": 8.0}

    def _default_rl_metrics(self) -> Dict[str, float]:
        return {"mean_reward": 0.0, "best_reward": 0.0, "reward_slope": 0.0}

    def run_eval(self, checkpoint_path: Optional[str] = None) -> BenchmarkReport:
        ckpt = checkpoint_path or str(self.cfg.rl.get("checkpoint_path", ""))
        if not ckpt:
            raise ValueError("No checkpoint path provided for evaluation.")
        ckpt_path = Path(ckpt)
        if ckpt_path.is_dir():
            report = self.engine.evaluate_from_artifacts(ckpt_path)
        else:
            report = self.engine.evaluate(
                checkpoint_path=str(ckpt_path),
                sim_metrics=self._default_metrics(),
                ftmo_metrics=self._default_ftmo_metrics(),
                rl_train_metrics=self._default_rl_metrics(),
            )
        self._persist_report(report)
        return report

    def run_multi_eval(self, checkpoints: List[str]) -> BenchmarkComparison:
        reports = [self.run_eval(cp) for cp in checkpoints]
        ranked = sorted(reports, key=lambda r: r.score, reverse=True)
        return BenchmarkComparison(best_checkpoint=ranked[0].checkpoint_path if ranked else "", ranked=ranked)

    def _persist_report(self, report: BenchmarkReport) -> None:
        out_root = Path(self.cfg.output.get("evaluation_root", "runs/eval"))
        out_root.mkdir(parents=True, exist_ok=True)
        name = Path(report.checkpoint_path).name.replace(".", "_")
        json_path = out_root / f"benchmark_{name}.json"
        txt_path = out_root / f"benchmark_{name}.txt"
        json_path.write_text(json.dumps(report.__dict__, indent=2))
        if self.cfg.output.get("save_txt", True):
            lines = [
                f"Checkpoint: {report.checkpoint_path}",
                f"Score: {report.score:.4f}",
                f"KPIs: {report.kpis}",
                f"FTMO: {report.ftmo}",
                f"RL Train: {report.rl_train}",
            ]
            txt_path.write_text("\n".join(lines))
        if self.cfg.output.get("save_html", False):
            html_cfg_data = self.cfg.output.get("html", {})
            html_cfg = HtmlReportConfig(
                title=html_cfg_data.get("title", "AFTS Benchmark Report"),
                include_equity_chart=html_cfg_data.get("include_equity_chart", True),
                include_drawdown_chart=html_cfg_data.get("include_drawdown_chart", True),
                include_daily_pnl_chart=html_cfg_data.get("include_daily_pnl_chart", True),
                include_ftmo_table=html_cfg_data.get("include_ftmo_table", True),
                include_rl_train_section=html_cfg_data.get("include_rl_train_section", True),
            )
            renderer = BenchmarkHtmlRenderer(html_cfg)
            run_dir = Path(report.checkpoint_path) if Path(report.checkpoint_path).is_dir() else out_root
            renderer.render_single(report, run_dir=run_dir)
