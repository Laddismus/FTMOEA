from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from afts_pro.rl.types import RLObsSpec, ActionSpec
from afts_pro.rl.risk_agent import RiskAgent, RiskAgentConfig
from afts_pro.rl.exit_agent import ExitAgent, ExitAgentConfig

logger = logging.getLogger(__name__)


@dataclass
class E2ESimConfig:
    config_path: str


@dataclass
class E2ERunResult:
    run_dir: Path
    num_trades: int
    equity_start: float
    equity_end: float
    has_rl_signals: bool
    files_present: Dict[str, bool]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_e2e_sim(config: E2ESimConfig) -> E2ERunResult:
    """
    Lightweight E2E SIM run that exercises RL inference and writes artifacts.
    """
    cfg = json.loads(Path(config.config_path).read_text()) if config.config_path.endswith(".json") else {}
    output_root = Path(cfg.get("output_root", "runs/e2e_acceptance"))
    _ensure_dir(output_root)
    run_dir = output_root / "run_e2e"
    _ensure_dir(run_dir)

    # Dummy agents (deterministic)
    obs_spec = RLObsSpec(shape=(4,), dtype="float32", as_dict=False)
    risk_agent = RiskAgent(RiskAgentConfig(exploration_epsilon=0.0), obs_spec, ActionSpec(action_type="continuous", shape=(1,)))
    exit_agent = ExitAgent(ExitAgentConfig(exploration_epsilon=0.0), obs_spec, ActionSpec(action_type="discrete", num_actions=6))
    obs = np.zeros(obs_spec.shape, dtype=np.float32)
    risk_pct = risk_agent.act(obs, deterministic=True)
    exit_action = exit_agent.act(obs, deterministic=True)

    # Build simple trades/equity data
    trades = pd.DataFrame(
        [
            {
                "trade_id": "e2e-1",
                "symbol": cfg.get("data", {}).get("symbol", "TEST_ASSET"),
                "pnl": 10.0,
                "risk_pct": risk_pct,
                "exit_action": exit_action,
            }
        ]
    )
    equity = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2024-01-01"), "equity": 10000.0},
            {"timestamp": pd.Timestamp("2024-01-02"), "equity": 10010.0},
        ]
    )
    metrics = {"num_trades": len(trades), "final_equity": float(equity["equity"].iloc[-1])}

    trades_path = run_dir / "trades.parquet"
    equity_path = run_dir / "equity_curve.parquet"
    metrics_path = run_dir / "metrics.json"
    trades.to_parquet(trades_path, index=False)
    equity.to_parquet(equity_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    files_present = {
        "trades": trades_path.exists(),
        "equity": equity_path.exists(),
        "metrics": metrics_path.exists(),
    }
    has_rl = "risk_pct" in trades.columns or "exit_action" in trades.columns
    result = E2ERunResult(
        run_dir=run_dir,
        num_trades=len(trades),
        equity_start=float(equity["equity"].iloc[0]),
        equity_end=float(equity["equity"].iloc[-1]),
        has_rl_signals=has_rl,
        files_present=files_present,
    )
    logger.info("E2E SIM run complete | trades=%d | equity_end=%.2f", result.num_trades, result.equity_end)
    return result
