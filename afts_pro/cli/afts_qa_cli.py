from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from afts_pro.core.qa_config import QAConfig
from afts_pro.core.qa_report import run_qa_suite, save_report

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="AFTS-PRO QA suite runner")
    parser.add_argument("--config", default="configs/qa/qa.yaml", help="Path to QA config YAML.")
    parser.add_argument("--output-dir", default="runs/qa", help="Directory to store QA reports.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg_data = yaml.safe_load(Path(args.config).read_text()) if Path(args.config).exists() else {}
    cfg = QAConfig(**cfg_data) if cfg_data else QAConfig()

    report = run_qa_suite(cfg)
    out_dir = Path(args.output_dir)
    json_path, txt_path = save_report(report, out_dir)

    logger.info("QA report written to %s and %s", json_path, txt_path)
    logger.info("QA overall status: %s", "PASSED" if report.all_passed else "FAILED")


if __name__ == "__main__":
    main()
