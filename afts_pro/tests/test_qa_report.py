from pathlib import Path

from afts_pro.core.qa_report import (
    QAReport,
    QASectionResult,
    QACheckResult,
    run_qa_suite,
    save_report,
)
from afts_pro.core.qa_config import QAConfig


def test_qa_report_all_sections_pass(monkeypatch):
    def _pass_section(name):
        return QASectionResult(name=name, checks=[QACheckResult(name="ok", passed=True)])

    monkeypatch.setattr("afts_pro.core.qa_report.run_e2e_section", lambda: _pass_section("e2e"))
    monkeypatch.setattr("afts_pro.core.qa_report.run_train_smoke_section", lambda tmp_root=None: _pass_section("train"))
    monkeypatch.setattr("afts_pro.core.qa_report.run_lab_smoke_section", lambda tmp_root=None: _pass_section("lab"))
    monkeypatch.setattr("afts_pro.core.qa_report.run_quant_smoke_section", lambda tmp_root=None: _pass_section("quant"))

    report = run_qa_suite(QAConfig(enable_pytest_smoke=False))
    assert report.all_passed is True
    assert {s.name for s in report.sections} == {"e2e", "train", "lab", "quant"}


def test_qa_report_fails_if_one_section_fails(monkeypatch):
    ok = QASectionResult(name="ok", checks=[QACheckResult(name="ok", passed=True)])
    bad = QASectionResult(name="bad", checks=[QACheckResult(name="bad_check", passed=False)])
    monkeypatch.setattr("afts_pro.core.qa_report.run_e2e_section", lambda: ok)
    monkeypatch.setattr("afts_pro.core.qa_report.run_train_smoke_section", lambda tmp_root=None: bad)
    monkeypatch.setattr("afts_pro.core.qa_report.run_lab_smoke_section", lambda tmp_root=None: ok)
    monkeypatch.setattr("afts_pro.core.qa_report.run_quant_smoke_section", lambda tmp_root=None: ok)
    report = run_qa_suite(QAConfig(enable_pytest_smoke=False))
    assert report.all_passed is False


def test_qa_cli_writes_report_files(tmp_path, monkeypatch):
    section = QASectionResult(name="ok", checks=[QACheckResult(name="ok", passed=True)])
    monkeypatch.setattr("afts_pro.core.qa_report.run_e2e_section", lambda: section)
    monkeypatch.setattr("afts_pro.core.qa_report.run_train_smoke_section", lambda tmp_root=None: section)
    monkeypatch.setattr("afts_pro.core.qa_report.run_lab_smoke_section", lambda tmp_root=None: section)
    monkeypatch.setattr("afts_pro.core.qa_report.run_quant_smoke_section", lambda tmp_root=None: section)
    report = run_qa_suite(QAConfig(enable_pytest_smoke=False))
    json_path, txt_path = save_report(report, tmp_path)
    assert json_path.exists()
    assert txt_path.exists()
