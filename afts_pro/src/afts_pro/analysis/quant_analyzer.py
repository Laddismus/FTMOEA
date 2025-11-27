from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from afts_pro.analysis.models import (
    DriftResult,
    MonteCarloResult,
    QuantConfig,
    RegimeResult,
    RollingKpiResult,
)
from afts_pro.runlogger.metrics import (
    compute_avg_win_loss,
    compute_max_drawdown,
    compute_profit_factor,
    compute_winrate,
)

logger = logging.getLogger(__name__)


def load_quant_config(path: str) -> QuantConfig:
    data = yaml.safe_load(Path(path).read_text())
    return QuantConfig(
        rolling=data.get("rolling", {}),
        monte_carlo=data.get("monte_carlo", {}),
        drift=data.get("drift", {}),
        regimes=data.get("regimes", {}),
        output=data.get("output", {}),
    )


class QuantAnalyzer:
    """
    Quantitative analyzer for run artifacts.
    """

    def __init__(self, config: QuantConfig) -> None:
        self.config = config

    def analyze_run(self, run_path: Path) -> Dict[str, Any]:
        run_path = Path(run_path)
        run_id = run_path.name
        equity_path = run_path / "equity_curve.parquet"
        trades_path = run_path / "trades.parquet"
        metrics_path = run_path / "metrics.json"
        equity_df = pd.read_parquet(equity_path) if equity_path.exists() else pd.DataFrame()
        trades_df = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

        output_root = Path(self.config.output.get("root_dir", "runs/analysis"))
        target_dir = output_root / run_id
        target_dir.mkdir(parents=True, exist_ok=True)

        summary: Dict[str, Any] = {"run_id": run_id}

        if not equity_df.empty and self.config.rolling:
            rolling_result = self.rolling_kpis(equity_df, trades_df)
            if self.config.output.get("save_rolling_kpis", True):
                path = target_dir / "rolling_kpis.parquet"
                rolling_result.df.to_parquet(path, index=False)
                summary["rolling_kpis_path"] = str(path)

        if not trades_df.empty and self.config.monte_carlo.get("enabled", True):
            mc_result = self.monte_carlo_analysis(trades_df)
            if self.config.output.get("save_monte_carlo", True):
                mc_path = target_dir / "monte_carlo.json"
                mc_path.write_text(json.dumps(mc_result.summary, indent=2))
            summary["monte_carlo"] = mc_result.summary

        if not equity_df.empty and self.config.drift.get("enabled", True):
            drift_result = self.detect_drift(equity_df)
            summary["n_drift_points"] = len(drift_result.drift_points)
            if self.config.output.get("save_drift", True):
                drift_path = target_dir / "drift.json"
                drift_path.write_text(json.dumps({"drift_points": [str(p) for p in drift_result.drift_points]}))

        if not equity_df.empty and self.config.regimes.get("enabled", True):
            regime_result = self.detect_regimes(equity_df)
            summary["regime_counts"] = dict(regime_result.regimes.value_counts())
            if self.config.output.get("save_regimes", True):
                regime_path = target_dir / "equity_with_regime.parquet"
                df = equity_df.copy()
                df["regime"] = regime_result.regimes.values
                df.to_parquet(regime_path, index=False)

        summary["metrics"] = metrics
        return summary

    def rolling_kpis(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> RollingKpiResult:
        cfg = self.config.rolling
        window = int(cfg.get("window_bars", 200))
        step = int(cfg.get("step_bars", 50))
        metrics_list = cfg.get("metrics", ["pf", "winrate", "mdd", "avg_r", "volatility"])
        records: List[Dict[str, Any]] = []
        eq_series = equity_df["equity"].reset_index(drop=True)
        for start in range(0, len(eq_series) - window + 1, step):
            end = start + window
            window_equity = eq_series.iloc[start:end]
            returns = window_equity.pct_change().dropna()
            rec: Dict[str, Any] = {
                "start_idx": start,
                "end_idx": end,
            }
            if "volatility" in metrics_list:
                rec["volatility"] = returns.std()
            if "mdd" in metrics_list:
                _, mdd_pct = compute_max_drawdown(
                    [type("ep", (), {"equity": e}) for e in window_equity]  # simple adapter
                )
                rec["mdd"] = mdd_pct
            # trade-based metrics if available
            if not trades_df.empty:
                window_trades = trades_df.iloc[start:end] if len(trades_df) >= end else trades_df
                pf = compute_profit_factor([type("tr", (), {"realized_pnl": v}) for v in window_trades.get("pnl", [])])  # type: ignore[arg-type]
                winrate = compute_winrate([type("tr", (), {"realized_pnl": v}) for v in window_trades.get("pnl", [])])  # type: ignore[arg-type]
                avg_win, avg_loss = compute_avg_win_loss([type("tr", (), {"realized_pnl": v}) for v in window_trades.get("pnl", [])])  # type: ignore[arg-type]
                if "pf" in metrics_list:
                    rec["pf"] = pf
                if "winrate" in metrics_list:
                    rec["winrate"] = winrate
                if "avg_r" in metrics_list and window_trades.get("r_multiple") is not None:
                    rec["avg_r"] = float(np.nan_to_num(window_trades["r_multiple"].mean()))
                elif "avg_r" in metrics_list and avg_win is not None and avg_loss is not None:
                    rec["avg_r"] = (avg_win or 0.0) / (avg_loss or 1.0)
            records.append(rec)
        df = pd.DataFrame(records)
        return RollingKpiResult(df=df, meta={"window": window, "step": step})

    def monte_carlo_analysis(self, trades_df: pd.DataFrame) -> MonteCarloResult:
        cfg = self.config.monte_carlo
        n_scenarios = int(cfg.get("n_scenarios", 1000))
        horizon = int(cfg.get("horizon_trades", 200))
        sampling = cfg.get("sampling", "bootstrap")
        trade_returns = trades_df["r_multiple"] if "r_multiple" in trades_df.columns else trades_df.get("pnl", pd.Series(dtype=float))
        trade_returns = trade_returns.fillna(0.0).to_numpy()
        if trade_returns.size == 0:
            return MonteCarloResult(summary={}, distribution=np.array([]))

        rng = np.random.default_rng()
        end_equity: List[float] = []
        for _ in range(n_scenarios):
            if sampling == "bootstrap":
                sampled = rng.choice(trade_returns, size=horizon, replace=True)
            else:
                sampled = rng.choice(trade_returns, size=horizon, replace=True)
            equity = 1.0
            for r in sampled:
                equity *= (1.0 + r)
            end_equity.append(equity)
        dist = np.array(end_equity)
        summary = {
            "mean": float(np.mean(dist)),
            "median": float(np.median(dist)),
            "p05": float(np.percentile(dist, 5)),
            "p95": float(np.percentile(dist, 95)),
            "worst": float(np.min(dist)),
        }
        return MonteCarloResult(summary=summary, distribution=dist)

    def detect_drift(self, equity_df: pd.DataFrame) -> DriftResult:
        cfg = self.config.drift
        if equity_df.empty:
            return DriftResult(drift_points=[])
        threshold_std = float(cfg.get("threshold_std", 2.5))
        equity = equity_df["equity"].astype(float)
        returns = equity.pct_change().dropna()
        mean = returns.mean()
        std = returns.std() or 1e-9
        pos_cusum = 0.0
        neg_cusum = 0.0
        drift_points: List[pd.Timestamp] = []
        for ts, ret in returns.items():
            pos_cusum = max(0.0, pos_cusum + ret - mean - threshold_std * std)
            neg_cusum = min(0.0, neg_cusum + ret - mean + threshold_std * std)
            if pos_cusum > 0:
                drift_points.append(ts)
                pos_cusum = 0.0
            if neg_cusum < 0:
                drift_points.append(ts)
                neg_cusum = 0.0
        return DriftResult(drift_points=drift_points, segments=[])

    def detect_regimes(self, equity_df: pd.DataFrame) -> RegimeResult:
        cfg = self.config.regimes
        n_regimes = int(cfg.get("n_regimes", 3))
        if equity_df.empty:
            return RegimeResult(regimes=pd.Series(dtype=int), meta={"n_regimes": n_regimes})
        window = int(cfg.get("window", 50))
        equity = equity_df["equity"].astype(float).reset_index(drop=True)
        rolling_return = equity.pct_change(periods=window).fillna(0.0)
        rolling_vol = equity.pct_change().rolling(window).std().fillna(0.0)
        score = rolling_return / (rolling_vol.replace(0, np.nan))
        score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        quantiles = np.linspace(0, 1, n_regimes + 1)
        thresholds = score.quantile(quantiles).to_numpy()
        regimes = np.digitize(score, thresholds[1:-1], right=True)
        return RegimeResult(regimes=pd.Series(regimes), meta={"thresholds": thresholds.tolist()})
