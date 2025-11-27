from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from afts_pro.exec.position_models import AccountState
from afts_pro.exec.position_manager import PositionEvent
from afts_pro.runlogger.metrics import build_metrics_snapshot
from afts_pro.runlogger.models import EquityPoint, MetricsSnapshot, RunMeta, TradeRecord

logger = logging.getLogger(__name__)


class RunLogger:
    """
    Passive, event-driven run logger that captures equity, trades and config snapshots.
    """

    def __init__(self, run_meta: RunMeta, config, project_root_path: Path) -> None:
        self.run_meta = run_meta
        self.config = config
        base_dir = Path(config.base_dir)
        if not base_dir.is_absolute():
            base_dir = project_root_path / base_dir
        self.run_dir = base_dir / run_meta.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trades: List[TradeRecord] = []
        self.equity_points: List[EquityPoint] = []
        self._max_equity: float = 0.0
        logger.info("RUNLOGGER_INIT | run_id=%s | dir=%s", run_meta.run_id, self.run_dir)

    def on_bar_equity_snapshot(
        self,
        bar_ts: datetime,
        account_state: AccountState,
        risk_meta: Optional[Dict] = None,
    ) -> None:
        equity = float(account_state.equity)
        balance = float(account_state.balance)
        unrealized = float(account_state.unrealized_pnl)
        realized_cum = float(account_state.realized_pnl)
        self._max_equity = max(self._max_equity, equity)
        max_equity = self._max_equity or equity
        dd_abs = max(max_equity - equity, 0.0)
        dd_pct = dd_abs / max_equity if max_equity else 0.0
        point = EquityPoint(
            timestamp=bar_ts,
            equity=equity,
            balance=balance,
            unrealized_pnl=unrealized,
            realized_pnl_cum=realized_cum,
            max_equity_to_date=max_equity,
            drawdown_abs=dd_abs,
            drawdown_pct=dd_pct,
        )
        self.equity_points.append(point)

    def on_trade_close(self, position_event: PositionEvent, ts: datetime, extra_tags: Optional[Dict] = None) -> None:
        if position_event.event_type.upper() != "CLOSED":
            return
        trade_id = uuid.uuid4().hex
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=position_event.symbol,
            side="unknown",
            entry_timestamp=ts,
            exit_timestamp=ts,
            entry_price=0.0,
            exit_price=0.0,
            size=0.0,
            realized_pnl=position_event.realized_pnl_delta,
            fees=0.0,
            tags=extra_tags or {},
        )
        self.trades.append(trade)

    def finalize_and_persist(self, global_config_snapshot: Dict) -> MetricsSnapshot:
        self.run_meta.finished_at = datetime.now(timezone.utc)
        patterns = self.config.filename_patterns
        include_map = self.config.include
        snapshot_payload = {"config": global_config_snapshot, "run_meta": self.run_meta.model_dump()}

        if include_map.get("config_snapshot", True):
            cfg_path = self.run_dir / patterns.get("config_snapshot", "config_used.yaml")
            with cfg_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(snapshot_payload, fh)

        if include_map.get("trades", True) and self.trades:
            trades_path = self.run_dir / patterns.get("trades", "trades.parquet")
            df = pd.DataFrame([t.model_dump() for t in self.trades])
            df.to_parquet(trades_path, index=False)

        if include_map.get("equity_curve", True) and self.equity_points:
            eq_path = self.run_dir / patterns.get("equity_curve", "equity_curve.parquet")
            df_eq = pd.DataFrame([p.model_dump() for p in self.equity_points])
            df_eq.to_parquet(eq_path, index=False)

        if include_map.get("positions", False):
            positions_path = self.run_dir / patterns.get("positions", "positions.parquet")
            df_pos = pd.DataFrame([])  # placeholder for future position history
            df_pos.to_parquet(positions_path, index=False)

        metrics = build_metrics_snapshot(self.trades, self.equity_points)
        if include_map.get("metrics", True):
            metrics_path = self.run_dir / patterns.get("metrics", "metrics.json")
            with metrics_path.open("w", encoding="utf-8") as fh:
                json.dump(metrics.model_dump(), fh, indent=2, default=str)

        return metrics
