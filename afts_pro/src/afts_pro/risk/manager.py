from __future__ import annotations

import logging
from datetime import datetime

from afts_pro.exec.position_models import AccountState
from afts_pro.risk.base_policy import BaseRiskPolicy, RiskDecision

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Thin wrapper that delegates to a risk policy.
    """

    def __init__(self, policy: BaseRiskPolicy, ftmo_engine=None, ftmo_plus_engine=None) -> None:
        self._policy = policy
        self.ftmo_engine = ftmo_engine
        self.ftmo_plus_engine = ftmo_plus_engine

    def on_bar(self, account_state: AccountState, ts: datetime) -> RiskDecision:
        decision = self._policy.evaluate(account_state=account_state, ts=ts)
        return decision

    def before_new_orders(self, account_state: AccountState, ts: datetime) -> RiskDecision:
        decision = self._policy.evaluate(account_state=account_state, ts=ts)
        if self.ftmo_engine is not None:
            self.ftmo_engine.on_new_equity(account_state.equity, account_state.realized_pnl, ts)
            decision.meta["ftmo_daily_loss_pct"] = self.ftmo_engine.current_daily_loss_pct()
            decision.meta["ftmo_overall_loss_pct"] = self.ftmo_engine.current_overall_loss_pct()
            if not self.ftmo_engine.can_open_new_trade():
                decision.allow_new_orders = False
                decision.reason = decision.reason or "FTMO_SAFETY"
                decision.meta["ftmo_blocked"] = True
            if self.ftmo_engine.should_force_close_all_positions():
                decision.meta["ftmo_force_flatten"] = True
        if self.ftmo_plus_engine is not None:
            self.ftmo_plus_engine.update_rolling_equity(ts, account_state.equity)
            roll_dd = self.ftmo_plus_engine.rolling_loss_pct()
            vel = self.ftmo_plus_engine.loss_velocity_pct_per_hour()
            sess_cfg = self.ftmo_plus_engine.session_for_time(ts)
            sess_dd = None
            if sess_cfg is not None:
                session_start_equity = getattr(account_state, "session_start_equity", account_state.equity)
                sess_dd = self.ftmo_plus_engine.session_loss_pct(session_start_equity, account_state.equity)
            self.ftmo_plus_engine.update_stage(
                ftmo_daily_loss_pct=self.ftmo_engine.current_daily_loss_pct() if self.ftmo_engine else 0.0,
                ftmo_overall_loss_pct=self.ftmo_engine.current_overall_loss_pct() if self.ftmo_engine else 0.0,
                rolling_loss_pct=roll_dd,
                loss_velocity_pct_per_hour=vel,
                session_loss_pct=sess_dd,
                num_recent_trades=getattr(account_state, "num_recent_trades", 0),
            )
            decision.meta["ftmo_plus_stage"] = self.ftmo_plus_engine.state.current_stage
            decision.meta["ftmo_plus_rolling_loss_pct"] = roll_dd
            decision.meta["ftmo_plus_loss_velocity"] = vel
            if sess_cfg is not None:
                decision.meta["ftmo_plus_session_name"] = sess_cfg.name
                decision.meta["ftmo_plus_session_loss_pct"] = sess_dd
            if not self.ftmo_plus_engine.exposure_allows_new_trade(
                open_trades_count=getattr(account_state, "open_trades_count", 0),
                total_open_risk_pct=getattr(account_state, "total_open_risk_pct", 0.0),
            ):
                decision.allow_new_orders = False
                decision.meta["ftmo_plus_blocked_exposure"] = True
            spread = getattr(account_state, "current_spread_pips", None)
            if spread is not None and not self.ftmo_plus_engine.spread_allows_new_trade(spread_pips=spread):
                decision.allow_new_orders = False
                decision.meta["ftmo_plus_blocked_spread"] = True
            if self.ftmo_plus_engine.is_in_news_window(ts):
                decision.allow_new_orders = False
                decision.meta["ftmo_plus_blocked_news"] = True
            if not self.ftmo_plus_engine.is_allowed_by_time_fence(ts):
                decision.allow_new_orders = False
                decision.meta["ftmo_plus_blocked_time"] = True
            if self.ftmo_engine is not None and self.ftmo_engine.state is not None:
                progress = self.ftmo_plus_engine.profit_target_progress_pct(
                    current_equity=account_state.equity, initial_equity=self.ftmo_engine.state.initial_equity
                )
                decision.meta["ftmo_plus_profit_progress_pct"] = progress
                if self.ftmo_plus_engine.is_profit_hard_lock(progress):
                    decision.allow_new_orders = False
                    decision.meta["ftmo_plus_blocked_profit_hard"] = True
                elif self.ftmo_plus_engine.is_profit_soft_lock(progress):
                    decision.meta["ftmo_plus_profit_soft_lock"] = True
            decision.meta["ftmo_plus_trading_days"] = getattr(account_state, "trading_days_count", 0)
            decision.meta["ftmo_plus_total_trades"] = getattr(account_state, "completed_trades_count", 0)
            trade_pnls = []
            if hasattr(account_state, "last_n_trade_pnls"):
                try:
                    trade_pnls = account_state.last_n_trade_pnls(self.ftmo_plus_engine.cfg.stability.kpi_window_trades)
                except Exception:  # pragma: no cover - defensive
                    trade_pnls = []
            kpis = self.ftmo_plus_engine.compute_stability_kpis(trade_pnls)
            decision.meta["ftmo_plus_pf"] = kpis.get("profit_factor", 0.0)
            decision.meta["ftmo_plus_winrate"] = kpis.get("winrate", 0.0)
            decision.meta["ftmo_plus_pnl_std"] = kpis.get("pnl_std", 0.0)
            if self.ftmo_plus_engine.is_stability_degraded(kpis):
                decision.meta["ftmo_plus_stability_degraded"] = True
            if self.ftmo_plus_engine.is_circuit_breaker_active(ts):
                decision.allow_new_orders = False
                decision.meta["ftmo_plus_blocked_circuit"] = True
            if self.ftmo_plus_engine.check_circuit_breaker(
                last_equity=getattr(account_state, "last_equity", account_state.equity),
                current_equity=account_state.equity,
                last_trade_slippage_pips=getattr(account_state, "last_trade_slippage_pips", None),
                now=ts,
            ):
                decision.meta["ftmo_force_flatten"] = True
                decision.meta["ftmo_plus_circuit_triggered"] = True
        return decision
