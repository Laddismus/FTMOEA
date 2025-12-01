from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from afts_pro.core.qa_config import QAConfig
from afts_pro.core.system_gate import GatePolicy, evaluate_gate, load_latest_report, run_gate_from_scratch
from afts_pro.core.qa_report import save_report

logger = logging.getLogger(__name__)


def _load_policy(path: str) -> GatePolicy:
    data = yaml.safe_load(Path(path).read_text()) if Path(path).exists() else {}
    return GatePolicy(
        required_sections=data.get("required_sections", GatePolicy().required_sections),
        optional_sections=data.get("optional_sections", []),
        allow_optional_fail=data.get("allow_optional_fail", True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AFTS-PRO System Gate CLI")
    parser.add_argument("--mode", choices=["run", "from-last"], default="run", help="Gate mode: run QA or use last report.")
    parser.add_argument("--qa-config", default="configs/qa/qa.yaml", help="QA config YAML.")
    parser.add_argument("--policy-config", default="configs/qa/gate_policy.yaml", help="Gate policy YAML.")
    parser.add_argument("--report-dir", default="runs/qa", help="Directory for QA reports.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    policy = _load_policy(args.policy_config)
    report_dir = Path(args.report_dir)

    if args.mode == "run":
        qa_cfg_data = yaml.safe_load(Path(args.qa_config).read_text()) if Path(args.qa_config).exists() else {}
        qa_cfg = QAConfig(**qa_cfg_data) if qa_cfg_data else QAConfig()
        decision = run_gate_from_scratch(qa_cfg, policy, output_dir=report_dir)
        ready = decision.ready_for_live
        status = "READY" if ready else "NOT READY"
        logger.info("Gate evaluation complete | status=%s | failed_sections=%s", status, decision.failed_sections)
        sys.exit(0 if ready else 1)
    else:
        report = load_latest_report(report_dir)
        if report is None:
            logger.error("No QA report found in %s", report_dir)
            sys.exit(1)
        decision = evaluate_gate(report, policy)
        status = "READY" if decision.ready_for_live else "NOT READY"
        logger.info("Gate evaluation from last report | status=%s | failed_sections=%s", status, decision.failed_sections)
        sys.exit(0 if decision.ready_for_live else 1)


if __name__ == "__main__":
    main()
