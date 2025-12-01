from pathlib import Path

from afts_pro.core.benchmark_html import BenchmarkHtmlRenderer, HtmlReportConfig
from afts_pro.core.benchmark_report import BenchmarkComparison, BenchmarkReport


def _report() -> BenchmarkReport:
    return BenchmarkReport(
        kpis={"profit_factor": 1.3, "winrate": 0.55, "mdd_pct": 4.2},
        ftmo={"ftmo_pass": True, "target_progress_pct": 65.0},
        rl_train={"mean_reward": 0.8, "best_reward": 1.0, "reward_slope": 0.1},
        score=0.73,
        checkpoint_path="ckpt",
    )


def test_render_single_creates_html_file(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Create simple equity.csv
    (run_dir / "equity.csv").write_text("0,100\n1,101\n2,99\n3,102\n")
    renderer = BenchmarkHtmlRenderer(HtmlReportConfig())
    out = renderer.render_single(_report(), run_dir)
    assert out.exists()
    content = out.read_text()
    assert "profit_factor" in content
    assert "FTMO" in content or "ftmo" in content
    assert "Score" in content


def test_render_single_handles_missing_equity_data(tmp_path):
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    renderer = BenchmarkHtmlRenderer(HtmlReportConfig())
    out = renderer.render_single(_report(), run_dir)
    assert out.exists()
    assert out.read_text()


def test_render_comparison_generates_table(tmp_path):
    comp = BenchmarkComparison(
        best_checkpoint="best",
        ranked=[
            _report(),
            BenchmarkReport(
                kpis={"profit_factor": 1.0, "winrate": 0.5, "mdd_pct": 5.0},
                ftmo={"ftmo_pass": False},
                rl_train={},
                score=0.5,
                checkpoint_path="other",
            ),
        ],
    )
    renderer = BenchmarkHtmlRenderer(HtmlReportConfig())
    out = tmp_path / "comp.html"
    renderer.render_comparison(comp, out)
    assert out.exists()
    assert "Checkpoint" in out.read_text()
