from pathlib import Path

from afts_pro.core.e2e_runner import E2ESimConfig, run_e2e_sim


def test_single_bar_sim_persists_minimal_artifacts(tmp_path):
    cfg_path = tmp_path / "sim_e2e.yaml"
    cfg_path.write_text("{}")
    result = run_e2e_sim(E2ESimConfig(config_path=str(cfg_path)))
    assert result.run_dir.exists()
    assert result.files_present.get("trades")
    assert result.files_present.get("equity")
