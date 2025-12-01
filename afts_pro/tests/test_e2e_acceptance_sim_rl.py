from pathlib import Path

from afts_pro.core.e2e_runner import E2ESimConfig, run_e2e_sim


def test_e2e_sim_run_produces_trades_and_equity(tmp_path):
    cfg_path = tmp_path / "sim_e2e.yaml"
    cfg_path.write_text("{}")
    result = run_e2e_sim(E2ESimConfig(config_path=str(cfg_path)))
    assert result.num_trades > 0
    assert result.equity_start > 0
    assert result.equity_end > 0
    assert result.files_present.get("trades") and result.files_present.get("equity")


def test_e2e_sim_run_uses_rl_fields(tmp_path):
    cfg_path = tmp_path / "sim_e2e.yaml"
    cfg_path.write_text("{}")
    result = run_e2e_sim(E2ESimConfig(config_path=str(cfg_path)))
    assert result.has_rl_signals is True


def test_e2e_sim_run_no_exceptions(tmp_path):
    cfg_path = tmp_path / "sim_e2e.yaml"
    cfg_path.write_text("{}")
    run_e2e_sim(E2ESimConfig(config_path=str(cfg_path)))
