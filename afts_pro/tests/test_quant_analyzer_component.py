import numpy as np
import pandas as pd

from afts_pro.analysis.quant_analyzer import QuantAnalyzer
from afts_pro.analysis.models import QuantConfig


def _config(tmp_path):
    return QuantConfig(
        rolling={"window_bars": 50, "step_bars": 25, "metrics": ["pf", "winrate", "mdd", "avg_r", "volatility"]},
        monte_carlo={"enabled": True, "n_scenarios": 10, "horizon_trades": 5, "sampling": "bootstrap"},
        drift={"enabled": True, "threshold_std": 2.5},
        regimes={"enabled": True, "n_regimes": 3, "window": 10},
        output={"root_dir": str(tmp_path), "save_rolling_kpis": False, "save_monte_carlo": False, "save_drift": False, "save_regimes": False},
    )


def _equity(n=200):
    ts = pd.date_range("2024-01-01", periods=n, freq="min")
    eq = np.linspace(100, 150, n) + np.random.normal(0, 1, size=n)
    return pd.DataFrame({"timestamp": ts, "equity": eq})


def _trades(n=50):
    rng = np.random.default_rng(0)
    pnl = rng.normal(0.1, 0.2, size=n)
    return pd.DataFrame({"pnl": pnl, "r_multiple": pnl})


def test_quant_analyzer_computes_rolling_kpis(tmp_path):
    qa = QuantAnalyzer(_config(tmp_path))
    result = qa.rolling_kpis(_equity(120), _trades(120))
    assert not result.df.empty


def test_quant_analyzer_detects_drift_via_cusum(tmp_path):
    qa = QuantAnalyzer(_config(tmp_path))
    equity = _equity(150)
    equity.loc[100:, "equity"] -= 20
    drift = qa.detect_drift(equity)
    assert len(drift.drift_points) >= 1


def test_quant_analyzer_labels_regimes(tmp_path):
    qa = QuantAnalyzer(_config(tmp_path))
    regimes = qa.detect_regimes(_equity(80))
    assert not regimes.regimes.empty
