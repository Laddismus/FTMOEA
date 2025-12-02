from pathlib import Path
from datetime import datetime, timezone

from afts_pro.core.system_gate import GatePolicy, evaluate_gate, run_gate_from_scratch
from afts_pro.core.qa_report import QAReport, QASectionResult, QACheckResult
from afts_pro.core.qa_config import QAConfig


def _make_report(section_status: dict) -> QAReport:
    sections = []
    for name, passed in section_status.items():
        sections.append(QASectionResult(name=name, checks=[QACheckResult(name="chk", passed=passed)]))
    return QAReport(sections=sections, generated_at=datetime.now(timezone.utc))


def test_evaluate_gate_all_required_sections_pass():
    report = _make_report({"e2e_sim_rl": True, "train_smoke": True, "lab_smoke": True, "quant_smoke": True})
    policy = GatePolicy()
    decision = evaluate_gate(report, policy)
    assert decision.ready_for_live is True
    assert decision.failed_sections == []


def test_evaluate_gate_missing_required_section_fails():
    report = _make_report({"e2e_sim_rl": True, "train_smoke": False, "lab_smoke": True, "quant_smoke": True})
    policy = GatePolicy()
    decision = evaluate_gate(report, policy)
    assert decision.ready_for_live is False
    assert "train_smoke" in decision.failed_sections


def test_evaluate_gate_optional_section_can_fail():
    report = _make_report({"e2e_sim_rl": True, "train_smoke": True, "lab_smoke": True, "quant_smoke": True, "pytest_smoke": False})
    policy = GatePolicy(optional_sections=["pytest_smoke"], allow_optional_fail=True)
    decision = evaluate_gate(report, policy)
    assert decision.ready_for_live is True


def test_run_gate_from_scratch_uses_qa_suite(monkeypatch, tmp_path):
    fake_report = _make_report({"e2e_sim_rl": True, "train_smoke": True, "lab_smoke": True, "quant_smoke": True})

    def fake_run(cfg):
        return fake_report

    monkeypatch.setattr("afts_pro.core.system_gate.run_qa_suite", fake_run)
    policy = GatePolicy()
    decision = run_gate_from_scratch(QAConfig(enable_pytest_smoke=False), policy, output_dir=tmp_path)
    assert decision.ready_for_live is True
    assert Path(tmp_path).exists()


def test_gate_cli_exit_codes(monkeypatch, tmp_path):
    class DummyDecision:
        def __init__(self, ready: bool):
            self.ready_for_live = ready
            self.failed_sections = []

    def fake_run_from_scratch(cfg, policy, output_dir):
        return DummyDecision(True)

    monkeypatch.setattr("afts_pro.core.system_gate.run_gate_from_scratch", fake_run_from_scratch)
    decision = run_gate_from_scratch(QAConfig(enable_pytest_smoke=False), GatePolicy(), tmp_path)
    assert decision.ready_for_live is True
