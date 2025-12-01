from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from afts_pro.core.qa_config import QAConfig
from afts_pro.core.qa_report import QAReport, run_qa_suite

logger = logging.getLogger(__name__)


@dataclass
class GatePolicy:
    required_sections: List[str] = field(
        default_factory=lambda: ["e2e_sim_rl", "train_smoke", "lab_smoke", "quant_smoke"]
    )
    optional_sections: List[str] = field(default_factory=list)
    allow_optional_fail: bool = True


@dataclass
class GateDecision:
    ready_for_live: bool
    failed_sections: List[str]
    report: QAReport


def evaluate_gate(report: QAReport, policy: Optional[GatePolicy] = None) -> GateDecision:
    policy = policy or GatePolicy()
    failed: List[str] = []
    section_map = {s.name: s for s in report.sections}

    for req in policy.required_sections:
        sec = section_map.get(req)
        if sec is None or not sec.passed:
            failed.append(req)

    if not policy.allow_optional_fail:
        for opt in policy.optional_sections:
            sec = section_map.get(opt)
            if sec is None or not sec.passed:
                failed.append(opt)

    ready = len(failed) == 0
    return GateDecision(ready_for_live=ready, failed_sections=failed, report=report)


def load_latest_report(report_dir: Path) -> Optional[QAReport]:
    if not report_dir.exists():
        return None
    reports = sorted(report_dir.glob("qa_report_*.json"), reverse=True)
    if not reports:
        return None
    latest = reports[0]
    data = json.loads(latest.read_text())
    sections = []
    from afts_pro.core.qa_report import QASectionResult, QACheckResult
    from datetime import datetime

    for sec in data.get("sections", []):
        checks = [QACheckResult(name=c["name"], passed=c["passed"], details=c.get("details", {})) for c in sec.get("checks", [])]
        sections.append(QASectionResult(name=sec["name"], checks=checks))
    generated_at = datetime.fromisoformat(data.get("generated_at"))
    return QAReport(sections=sections, generated_at=generated_at)


def run_gate_from_scratch(qa_config: QAConfig, policy: GatePolicy, output_dir: Path) -> GateDecision:
    report = run_qa_suite(qa_config)
    from afts_pro.core.qa_report import save_report

    save_report(report, output_dir)
    return evaluate_gate(report, policy)
