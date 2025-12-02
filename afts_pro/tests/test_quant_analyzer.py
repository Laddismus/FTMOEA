import numpy as np
import pandas as pd
from pathlib import Path

from afts_pro.analysis.quant_analyzer import QuantAnalyzer, load_quant_config
from afts_pro.analysis.models import QuantConfig


def _synthetic_equity(n: int = 1000) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="min")
    trend = np.linspace(0, 1.0, n)
    noise = np.random.normal(0, 0.01, size=n)
    eq = 1.0 + trend + noise
    # inject drawdown
    eq[500:600] -= 0.2
    return pd.DataFrame({"timestamp": ts, "equity": eq})


def _synthetic_trades(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    r = rng.normal(0.003, 0.01, size=n)
    return pd.DataFrame({"pnl": r, "r_multiple": r})


def _config(tmp_path: Path) -> QuantConfig:
    cfg_path = tmp_path / "quant.yaml"
    cfg_path.write_text(
        """
rolling:
  window_bars: 100
  step_bars: 50
  metrics: ["pf","winrate","mdd","avg_r","volatility"]
monte_carlo:
  enabled: true
  n_scenarios: 100
  horizon_trades: 50
  sampling: "bootstrap"
drift:
  enabled: true
  threshold_std: 2.5
regimes:
  enabled: true
  n_regimes: 3
  window: 30
output:
  root_dir: "runs/analysis"
  save_rolling_kpis: false
  save_monte_carlo: false
  save_drift: false
  save_regimes: false
        """
    )
    return load_quant_config(str(cfg_path))


def test_rolling_kpis_shapes(tmp_path):
    cfg = _config(tmp_path)
    qa = QuantAnalyzer(cfg)
    equity = _synthetic_equity(300)
    trades = _synthetic_trades(300)
    result = qa.rolling_kpis(equity, trades)
    assert not result.df.empty
    assert {"pf", "winrate"}.issubset(result.df.columns)


def test_monte_carlo_basic_stats(tmp_path):
    cfg = _config(tmp_path)
    qa = QuantAnalyzer(cfg)
    trades = _synthetic_trades(200)
    mc = qa.monte_carlo_analysis(trades)
    assert mc.summary["p05"] <= mc.summary["median"] <= mc.summary["p95"]


def test_drift_detection_cusum(tmp_path):
    cfg = _config(tmp_path)
    qa = QuantAnalyzer(cfg)
    equity = _synthetic_equity(800)
    drift = qa.detect_drift(equity)
    assert len(drift.drift_points) >= 1


def test_regime_detection_labels(tmp_path):
    cfg = _config(tmp_path)
    qa = QuantAnalyzer(cfg)
    equity = _synthetic_equity(200)
    regimes = qa.detect_regimes(equity)
    labels = regimes.regimes.unique()
    assert len(labels) <= cfg.regimes.get("n_regimes", 3)
