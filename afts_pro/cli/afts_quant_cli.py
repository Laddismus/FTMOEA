from __future__ import annotations

import argparse
import logging
from pathlib import Path

from afts_pro.analysis.quant_analyzer import QuantAnalyzer, load_quant_config

logger = logging.getLogger(__name__)


def _build_analyzer(cfg_path: str) -> QuantAnalyzer:
    config = load_quant_config(cfg_path)
    return QuantAnalyzer(config)


def analyze_run_cmd(args: argparse.Namespace) -> None:
    analyzer = _build_analyzer(args.config)
    summary = analyzer.analyze_run(Path(args.run_path))
    logger.info("Quant analysis complete | summary=%s", summary)


def analyze_lab_sweep_cmd(args: argparse.Namespace) -> None:
    analyzer = _build_analyzer(args.config)
    sweep_path = Path(args.sweep_path)
    for exp_dir in sweep_path.glob("experiments/*"):
        if exp_dir.is_dir():
            summary = analyzer.analyze_run(exp_dir)
            logger.info("Analyzed experiment %s | summary=%s", exp_dir.name, summary)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="AFTS-PRO Quant Analyzer CLI")
    subparsers = parser.add_subparsers(dest="command")

    run_cmd = subparsers.add_parser("analyze-run", help="Analyze a single run directory.")
    run_cmd.add_argument("--config", default="configs/analysis/quant.yaml", help="Quant config YAML.")
    run_cmd.add_argument("--run-path", required=True, help="Path to run directory.")

    sweep_cmd = subparsers.add_parser("analyze-lab-sweep", help="Analyze all experiments in a LAB sweep directory.")
    sweep_cmd.add_argument("--config", default="configs/analysis/quant.yaml", help="Quant config YAML.")
    sweep_cmd.add_argument("--sweep-path", required=True, help="Path to LAB sweep directory.")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    if args.command == "analyze-run":
        analyze_run_cmd(args)
    elif args.command == "analyze-lab-sweep":
        analyze_lab_sweep_cmd(args)


if __name__ == "__main__":
    main()
