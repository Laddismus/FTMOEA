from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from afts_pro.lab.runner import LabRunner, load_lab_config
from afts_pro.lab.models import LabExperiment, LabSweepDefinition, RunResult

logger = logging.getLogger(__name__)


class BasicSimApi:
    """
    Minimal stub SIM backtest API used by the CLI to allow smoke runs.
    """

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def run_backtest(self, profile_name: str, overrides: Dict[str, Any] | None = None, seed: int | None = None) -> RunResult:
        run_id = uuid.uuid4().hex[:8]
        run_path = self.root_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        metrics = {"pf": 1.0, "winrate": 0.5, "mdd": 0.1, "trades": 10}
        metrics_path = run_path / "metrics.json"
        metrics_path.write_text(str(metrics))
        return RunResult(run_id=run_id, run_path=str(run_path), metrics=metrics)


def _parse_params(param_str: str | None) -> Dict[str, Any]:
    if not param_str:
        return {}
    params: Dict[str, Any] = {}
    for part in param_str.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                v_cast: Any = float(v)
            except ValueError:
                v_cast = v
            params[k.strip()] = v_cast
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="AFTS-PRO LAB CLI")
    subparsers = parser.add_subparsers(dest="command")

    run_once = subparsers.add_parser("run-once", help="Run a single LAB experiment.")
    run_once.add_argument("--config", required=True, help="Path to lab config YAML.")
    run_once.add_argument("--profile", required=False, help="Override base profile.")
    run_once.add_argument("--params", required=False, help="Comma-separated key=value overrides.")

    sweep = subparsers.add_parser("sweep", help="Run a sweep from lab config.")
    sweep.add_argument("--config", required=True, help="Path to lab config YAML.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not args.command:
        parser.print_help()
        return

    cfg = load_lab_config(args.config)
    output_cfg = cfg.get("output", {})
    sim_api = BasicSimApi(Path(output_cfg.get("root_dir", "runs/lab/raw_runs")))
    runner = LabRunner(cfg, sim_api)

    if args.command == "run-once":
        params = _parse_params(getattr(args, "params", None))
        exp = LabExperiment(
            id=uuid.uuid4().hex[:8],
            name="cli_run",
            mode=cfg.get("default_mode", "strategy_backtest"),
            base_profile=args.profile or cfg.get("base_profile", "sim"),
            params=params,
            seed=None,
            meta={},
        )
        result = runner.run_experiment(exp, sweep_id=None)
        logger.info("LAB run complete | experiment_id=%s | run_id=%s", exp.id, result.run_id)
    elif args.command == "sweep":
        sweep_cfg = cfg.get("sweep", {})
        sweep_def = LabSweepDefinition(
            id=uuid.uuid4().hex[:6],
            type=sweep_cfg.get("type", "grid"),
            params=sweep_cfg.get("params", {}),
            max_experiments=sweep_cfg.get("max_experiments"),
            seed=sweep_cfg.get("seed"),
        )
        results = runner.run_sweep(sweep_def)
        logger.info("LAB sweep complete | experiments=%d", len(results))


if __name__ == "__main__":
    main()
