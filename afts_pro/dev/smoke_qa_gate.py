from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from afts_pro.core.qa_config import QAConfig  # noqa: E402
from afts_pro.core.qa_report import run_qa_suite  # noqa: E402
from afts_pro.core.system_gate import GatePolicy, evaluate_gate  # noqa: E402

QA_CFG_PATH = Path("configs/qa/qa.yaml")
GATE_POLICY_PATH = Path("configs/qa/gate_policy.yaml")
OUTPUT_ROOT = Path("artifacts/dev_smoke_qa")


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    return yaml.safe_load(path.read_text()) or {}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    qa_cfg_data = load_yaml(QA_CFG_PATH)
    gate_cfg_data = load_yaml(GATE_POLICY_PATH)
    qa_cfg = QAConfig(**qa_cfg_data) if qa_cfg_data else QAConfig()
    gate_policy = GatePolicy(**gate_cfg_data) if gate_cfg_data else GatePolicy()

    logging.info("[QA-SMOKE] Starting QA suite")
    report = run_qa_suite(qa_cfg)
    report_path = OUTPUT_ROOT / "qa_report.json"
    report_path.write_text(json.dumps(report, default=str, indent=2))
    logging.info("[QA-SMOKE] QA suite finished | all_passed=%s | sections=%d", report.all_passed, len(report.sections))

    logging.info("[GATE] Evaluating gate policy")
    decision = evaluate_gate(report, gate_policy)
    gate_json = OUTPUT_ROOT / "gate_decision.json"
    gate_json.write_text(json.dumps(decision.__dict__, default=str, indent=2))
    logging.info("[GATE] ready_for_live=%s | failed_sections=%s", decision.ready_for_live, decision.failed_sections)

    if not decision.ready_for_live:
        logging.error("[GATE] Gate failed.")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
