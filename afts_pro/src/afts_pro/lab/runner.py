from __future__ import annotations

import itertools
import logging
import random
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import yaml

from afts_pro.lab.kpi_matrix import build_kpi_matrix, save_kpi_matrix
from afts_pro.lab.models import LabExperiment, LabResult, LabSweepDefinition, RunResult

logger = logging.getLogger(__name__)


def load_lab_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _generate_grid(params: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(params.keys())
    value_lists = [list(v) for v in params.values()]
    combinations = []
    for values in itertools.product(*value_lists):
        combo = {k: v for k, v in zip(keys, values)}
        combinations.append(combo)
    return combinations


def _generate_random(params: Dict[str, Iterable[Any]], max_experiments: int, seed: int | None = None) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    keys = list(params.keys())
    combinations: List[Dict[str, Any]] = []
    for _ in range(max_experiments):
        combo = {}
        for k in keys:
            values = list(params[k])
            combo[k] = rng.choice(values)
        combinations.append(combo)
    return combinations


class LabRunner:
    """
    Orchestrates LAB experiments and sweeps on top of a SIM backtest API.
    """

    def __init__(self, lab_config: Dict[str, Any], sim_api) -> None:
        self.lab_config = lab_config
        self.sim_api = sim_api
        output_cfg = lab_config.get("output", {})
        root_dir = output_cfg.get("root_dir", "runs/lab")
        self.lab_root = Path(root_dir)
        self.lab_root.mkdir(parents=True, exist_ok=True)

    def _build_experiment_dir(self, experiment: LabExperiment, sweep_id: str | None) -> Path:
        base = self.lab_root / (sweep_id or "single") / "experiments" / experiment.id
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _write_experiment_snapshot(self, path: Path, experiment: LabExperiment) -> None:
        snapshot = {
            "id": experiment.id,
            "name": experiment.name,
            "mode": experiment.mode,
            "base_profile": experiment.base_profile,
            "params": experiment.params,
            "seed": experiment.seed,
            "meta": experiment.meta,
        }
        with (path / "experiment_config.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(snapshot, fh)

    def run_experiment(self, experiment: LabExperiment, sweep_id: str | None = None) -> LabResult:
        """
        Run a single experiment via the provided SIM API.
        """
        run_result: RunResult = self.sim_api.run_backtest(experiment.base_profile, overrides=experiment.params, seed=experiment.seed)
        exp_dir = self._build_experiment_dir(experiment, sweep_id)
        self._write_experiment_snapshot(exp_dir, experiment)
        metrics_path = exp_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(run_result.metrics, fh)

        # Optionally copy equity/trades if available
        output_cfg = self.lab_config.get("output", {})
        if output_cfg.get("save_equity", False):
            src = Path(run_result.run_path) / "equity_curve.parquet"
            if src.exists():
                dst = exp_dir / "equity_curve.parquet"
                dst.write_bytes(src.read_bytes())
        if output_cfg.get("save_trades", False):
            src_trades = Path(run_result.run_path) / "trades.parquet"
            if src_trades.exists():
                dst_trades = exp_dir / "trades.parquet"
                dst_trades.write_bytes(src_trades.read_bytes())

        result = LabResult(
            experiment_id=experiment.id,
            run_id=run_result.run_id,
            run_path=run_result.run_path,
            metrics=run_result.metrics,
            params=experiment.params,
            meta=experiment.meta,
        )
        return result

    def run_sweep(self, sweep: LabSweepDefinition) -> List[LabResult]:
        """
        Execute a sweep and return LabResults.
        """
        if sweep.type == "grid":
            combos = _generate_grid(sweep.params)
        else:
            max_exp = sweep.max_experiments or 0
            combos = _generate_random(sweep.params, max_exp, seed=sweep.seed)
        if sweep.max_experiments:
            combos = combos[: sweep.max_experiments]
        results: List[LabResult] = []
        for combo in combos:
            exp = LabExperiment(
                id=uuid.uuid4().hex[:8],
                name=f"exp_{len(results)}",
                mode=self.lab_config.get("default_mode", "strategy_backtest"),
                base_profile=self.lab_config.get("base_profile", "sim"),
                params=combo,
                seed=sweep.seed,
                meta={"sweep_id": sweep.id},
            )
            res = self.run_experiment(exp, sweep_id=sweep.id)
            results.append(res)

        metrics_names = self.lab_config.get("metrics", [])
        output_cfg = self.lab_config.get("output", {})
        if output_cfg.get("save_kpi_matrix", True) and results and metrics_names:
            df = build_kpi_matrix(results, metrics_names)
            kpi_path = self.lab_root / sweep.id / "kpis.parquet"
            save_kpi_matrix(df, kpi_path, fmt="parquet")
        return results
