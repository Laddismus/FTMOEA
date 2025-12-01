from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from afts_pro.core.benchmark_report import BenchmarkReport


class BenchmarkEngine:
    """
    Lightweight benchmark calculator for SIM/RL results.
    """

    def __init__(self, scoring_weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = scoring_weights or {
            "pf": 1.0,
            "winrate": 1.0,
            "dd_penalty": 1.0,
            "ftmo_pass": 1.0,
            "reward_slope": 0.5,
        }

    def _load_json_if_exists(self, path: Path) -> Dict:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                return {}
        return {}

    def evaluate(
        self,
        *,
        checkpoint_path: str,
        sim_metrics: Dict[str, float],
        ftmo_metrics: Dict[str, float | bool],
        rl_train_metrics: Dict[str, float],
    ) -> BenchmarkReport:
        pf = float(sim_metrics.get("profit_factor", 0.0))
        winrate = float(sim_metrics.get("winrate", 0.0))
        mdd_pct = float(sim_metrics.get("mdd_pct", 0.0))
        reward_slope = float(rl_train_metrics.get("reward_slope", 0.0))
        ftmo_pass = bool(ftmo_metrics.get("ftmo_pass", False))

        score = (
            self.weights["pf"] * pf
            + self.weights["winrate"] * winrate
            - self.weights["dd_penalty"] * (mdd_pct / 100.0)
            + self.weights["ftmo_pass"] * (1.0 if ftmo_pass else 0.0)
            + self.weights["reward_slope"] * reward_slope
        )
        return BenchmarkReport(
            kpis=sim_metrics,
            ftmo=ftmo_metrics,
            rl_train=rl_train_metrics,
            score=score,
            checkpoint_path=checkpoint_path,
            comments=[],
        )

    def evaluate_from_artifacts(self, checkpoint_dir: Path) -> BenchmarkReport:
        sim_metrics = self._load_json_if_exists(checkpoint_dir / "metrics.json")
        ftmo_metrics = self._load_json_if_exists(checkpoint_dir / "ftmo_metrics.json")
        rl_train_metrics = self._load_json_if_exists(checkpoint_dir / "train_summary.json")
        return self.evaluate(
            checkpoint_path=str(checkpoint_dir),
            sim_metrics=sim_metrics,
            ftmo_metrics=ftmo_metrics,
            rl_train_metrics=rl_train_metrics,
        )
