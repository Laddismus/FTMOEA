import json
from pathlib import Path

from afts_pro.core.benchmark_report import BenchmarkReport
from afts_pro.core.model_selection import ModelSelectionConfig, ModelSelectionCriteria, ModelSelector


def _report(score: float, pf: float, mdd: float, win: float, ftmo_pass=True) -> BenchmarkReport:
    return BenchmarkReport(
        kpis={"profit_factor": pf, "mdd_pct": mdd, "winrate": win},
        ftmo={"ftmo_pass": ftmo_pass, "daily_dd_pass": ftmo_pass, "overall_dd_pass": ftmo_pass},
        rl_train={},
        score=score,
        checkpoint_path="ckpt",
    )


def test_filter_reports_applies_criteria():
    crit = ModelSelectionCriteria(min_profit_factor=1.2, max_drawdown_pct=-8.0, min_winrate=0.5, min_score=0.1)
    cfg = ModelSelectionConfig(
        eval_root=".", agent_type="risk", criteria=crit, promotion_root=".", promotion_tag="tag"
    )
    selector = ModelSelector(cfg)
    reports = [
        _report(0.2, 1.3, -7.0, 0.6),
        _report(0.2, 1.0, -7.0, 0.6),  # pf too low
        _report(0.05, 1.3, -7.0, 0.6),  # score too low
    ]
    filtered = selector.filter_reports(reports)
    assert len(filtered) == 1


def test_rank_reports_sorts_by_score_then_pf_then_dd():
    crit = ModelSelectionCriteria(min_profit_factor=0.0, max_drawdown_pct=-100.0, min_winrate=0.0, min_score=0.0)
    cfg = ModelSelectionConfig(eval_root=".", agent_type="risk", criteria=crit, promotion_root=".", promotion_tag="tag")
    selector = ModelSelector(cfg)
    r1 = _report(0.4, 1.2, -6.0, 0.5)
    r2 = _report(0.4, 1.3, -5.0, 0.5)
    r3 = _report(0.3, 1.5, -4.0, 0.5)
    ranked = selector.rank_reports([r1, r2, r3])
    assert ranked[0] is r2  # higher pf wins when score tie


def test_select_best_returns_none_if_no_model_passes(tmp_path):
    crit = ModelSelectionCriteria(min_profit_factor=2.0)
    cfg = ModelSelectionConfig(
        eval_root=str(tmp_path), agent_type="risk", criteria=crit, promotion_root=".", promotion_tag="tag"
    )
    selector = ModelSelector(cfg)
    assert selector.select_best() is None


def test_promote_creates_pointer_file_and_meta(tmp_path):
    crit = ModelSelectionCriteria(min_profit_factor=0.0, max_drawdown_pct=-100.0, min_winrate=0.0, min_score=0.0)
    cfg = ModelSelectionConfig(
        eval_root=".",
        agent_type="risk",
        criteria=crit,
        promotion_root=str(tmp_path),
        promotion_tag="test_tag",
        copy_checkpoint=False,
        pointer_filename="CURRENT.txt",
    )
    selector = ModelSelector(cfg)
    report = _report(0.5, 1.0, -1.0, 0.5)
    target = selector.promote(report, dry_run=False)
    pointer = Path(target) / "CURRENT.txt"
    meta = Path(target) / "selection_info.json"
    assert pointer.exists()
    assert meta.exists()
    meta_data = json.loads(meta.read_text())
    assert meta_data["checkpoint"] == report.checkpoint_path


def test_promote_dry_run_does_not_create_files(tmp_path):
    crit = ModelSelectionCriteria(min_profit_factor=0.0, max_drawdown_pct=-100.0, min_winrate=0.0, min_score=0.0)
    cfg = ModelSelectionConfig(
        eval_root=".",
        agent_type="risk",
        criteria=crit,
        promotion_root=str(tmp_path),
        promotion_tag="test_tag",
        copy_checkpoint=False,
        pointer_filename="CURRENT.txt",
    )
    selector = ModelSelector(cfg)
    report = _report(0.5, 1.0, -1.0, 0.5)
    target = selector.promote(report, dry_run=True)
    pointer = Path(target) / "CURRENT.txt"
    assert not pointer.exists()
