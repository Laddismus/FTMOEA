from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time, timezone
from statistics import pstdev
from typing import List, Literal, Optional, Tuple

RiskStage = Literal[0, 1, 2]  # 0=normal, 1=reduced, 2=freeze


@dataclass
class SessionRiskConfig:
    name: str
    start_time: str  # "HH:MM"
    end_time: str
    max_session_loss_pct: float
    soft_factor: float = 0.7


@dataclass
class RollingRiskConfig:
    window_minutes: int = 60
    max_rolling_loss_pct: float = 1.5


@dataclass
class LossVelocityConfig:
    dd_fast_threshold_pct_per_hour: float = 4.0


@dataclass
class RiskStageConfig:
    stage0_risk_mult: float = 1.0
    stage1_risk_mult: float = 0.5
    stage2_risk_mult: float = 0.0
    stage0_max_risk_pct: float = 1.0
    stage1_max_risk_pct: float = 0.5
    stage2_max_risk_pct: float = 0.0
    new_stage_min_trades: int = 5
    cooldown_minutes: int = 60


@dataclass
class ExposureCapsConfig:
    max_open_trades: int = 3
    max_total_risk_pct: float = 2.0


@dataclass
class SpreadGuardConfig:
    max_spread_pips: float = 0.8
    enabled: bool = True


@dataclass
class NewsWindowConfig:
    name: str
    start_datetime: str  # ISO string
    end_datetime: str


@dataclass
class TimeFenceConfig:
    name: str
    daily_start_time: str  # "HH:MM"
    daily_end_time: str
    mode: Literal["allow_only", "block"] = "allow_only"


@dataclass
class ProfitTargetConfig:
    target_pct: float = 10.0
    soft_lock_pct: float = 80.0
    hard_lock_pct: float = 100.0
    allow_small_maintenance_trades: bool = False


@dataclass
class ActivityConfig:
    min_trading_days: int = 10
    min_trades_total: int = 10
    min_trades_per_week: int = 3


@dataclass
class PerformanceStabilityConfig:
    kpi_window_trades: int = 20
    min_profit_factor: float = 1.1
    min_winrate: float = 0.45
    max_pnl_std_multiple: float = 3.0


@dataclass
class CircuitBreakerConfig:
    max_instant_loss_pct: float = 2.0
    max_slippage_pips: float = 1.5
    freeze_minutes: int = 30


@dataclass
class FtmoPlusConfig:
    sessions: List[SessionRiskConfig]
    rolling: RollingRiskConfig
    loss_velocity: LossVelocityConfig
    stages: RiskStageConfig
    exposure_caps: ExposureCapsConfig
    spread_guard: SpreadGuardConfig
    news_windows: List[NewsWindowConfig] = field(default_factory=list)
    time_fences: List[TimeFenceConfig] = field(default_factory=list)
    profit_target: ProfitTargetConfig = field(default_factory=ProfitTargetConfig)
    activity: ActivityConfig = field(default_factory=ActivityConfig)
    stability: PerformanceStabilityConfig = field(default_factory=PerformanceStabilityConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)


@dataclass
class FtmoPlusState:
    current_stage: RiskStage = 0
    last_stage_change: Optional[datetime] = None
    rolling_window: List[Tuple[datetime, float]] = None
    circuit_freeze_until: Optional[datetime] = None


class FtmoPlusEngine:
    def __init__(self, cfg: FtmoPlusConfig):
        self.cfg = cfg
        self.state = FtmoPlusState(current_stage=0, last_stage_change=None, rolling_window=[])

    def update_rolling_equity(self, now: datetime, equity: float) -> None:
        window = self.state.rolling_window
        window.append((now, equity))
        cutoff = now - timedelta(minutes=self.cfg.rolling.window_minutes)
        self.state.rolling_window = [(ts, eq) for ts, eq in window if ts >= cutoff]

    def rolling_loss_pct(self) -> float:
        if not self.state.rolling_window:
            return 0.0
        max_eq = max(eq for _, eq in self.state.rolling_window)
        current_eq = self.state.rolling_window[-1][1]
        if max_eq == 0:
            return 0.0
        return max(0.0, (max_eq - current_eq) / max_eq * 100.0)

    def loss_velocity_pct_per_hour(self) -> float:
        window = self.state.rolling_window
        if len(window) < 2:
            return 0.0
        start_ts, start_eq = window[0]
        end_ts, end_eq = window[-1]
        delta_hours = (end_ts - start_ts).total_seconds() / 3600.0
        if delta_hours <= 0 or start_eq == 0:
            return 0.0
        return ((start_eq - end_eq) / start_eq) / delta_hours * 100.0

    def session_for_time(self, now: datetime) -> Optional[SessionRiskConfig]:
        t = now.time()
        for sess in self.cfg.sessions:
            start_t = time.fromisoformat(sess.start_time)
            end_t = time.fromisoformat(sess.end_time)
            if start_t <= t <= end_t:
                return sess
        return None

    def session_loss_pct(self, session_start_equity: float, current_equity: float) -> float:
        if session_start_equity == 0:
            return 0.0
        return max(0.0, (session_start_equity - current_equity) / session_start_equity * 100.0)

    def update_stage(
        self,
        *,
        ftmo_daily_loss_pct: float,
        ftmo_overall_loss_pct: float,
        rolling_loss_pct: float,
        loss_velocity_pct_per_hour: float,
        session_loss_pct: float | None,
        num_recent_trades: int,
    ) -> None:
        stage = self.state.current_stage
        now = datetime.now(timezone.utc)
        # Escalate conditions
        escalate_to_stage2 = (
            rolling_loss_pct > self.cfg.rolling.max_rolling_loss_pct
            or loss_velocity_pct_per_hour > self.cfg.loss_velocity.dd_fast_threshold_pct_per_hour
            or ftmo_daily_loss_pct >= 0.9 * self.cfg.stages.stage1_max_risk_pct * 100  # rough heuristic
        )
        if session_loss_pct is not None:
            for sess in self.cfg.sessions:
                if session_loss_pct >= sess.max_session_loss_pct:
                    escalate_to_stage2 = True
        if escalate_to_stage2:
            self.state.current_stage = 2
            self.state.last_stage_change = now
            return
        if stage == 2:
            return
        escalate_to_stage1 = rolling_loss_pct > (0.5 * self.cfg.rolling.max_rolling_loss_pct) or loss_velocity_pct_per_hour > (
            0.5 * self.cfg.loss_velocity.dd_fast_threshold_pct_per_hour
        )
        if escalate_to_stage1 and stage < 1:
            self.state.current_stage = 1
            self.state.last_stage_change = now
            return
        # De-escalate if cooldown passed and conditions mild
        if stage > 0 and self.state.last_stage_change:
            cooldown = timedelta(minutes=self.cfg.stages.cooldown_minutes)
            if now - self.state.last_stage_change > cooldown and num_recent_trades >= self.cfg.stages.new_stage_min_trades:
                self.state.current_stage = stage - 1
                self.state.last_stage_change = now

    def current_stage_risk_mult(self) -> float:
        stage = self.state.current_stage
        cfg = self.cfg.stages
        if stage == 0:
            return cfg.stage0_risk_mult
        if stage == 1:
            return cfg.stage1_risk_mult
        return cfg.stage2_risk_mult

    def current_stage_max_risk_pct(self) -> float:
        stage = self.state.current_stage
        cfg = self.cfg.stages
        if stage == 0:
            return cfg.stage0_max_risk_pct
        if stage == 1:
            return cfg.stage1_max_risk_pct
        return cfg.stage2_max_risk_pct

    def exposure_allows_new_trade(self, *, open_trades_count: int, total_open_risk_pct: float) -> bool:
        if open_trades_count >= self.cfg.exposure_caps.max_open_trades:
            return False
        if total_open_risk_pct >= self.cfg.exposure_caps.max_total_risk_pct:
            return False
        return True

    def spread_allows_new_trade(self, spread_pips: float) -> bool:
        if not self.cfg.spread_guard.enabled:
            return True
        return spread_pips <= self.cfg.spread_guard.max_spread_pips

    def is_in_news_window(self, now: datetime) -> bool:
        for win in self.cfg.news_windows:
            try:
                start = datetime.fromisoformat(win.start_datetime)
                end = datetime.fromisoformat(win.end_datetime)
            except ValueError:
                continue
            if start <= now <= end:
                return True
        return False

    def is_allowed_by_time_fence(self, now: datetime) -> bool:
        if not self.cfg.time_fences:
            return True
        t = now.time()
        allowed_any = False
        for fence in self.cfg.time_fences:
            start_t = time.fromisoformat(fence.daily_start_time)
            end_t = time.fromisoformat(fence.daily_end_time)
            inside = start_t <= t <= end_t
            if fence.mode == "allow_only" and inside:
                allowed_any = True
            if fence.mode == "block" and inside:
                return False
        allow_only_fences = [f for f in self.cfg.time_fences if f.mode == "allow_only"]
        if allow_only_fences:
            return allowed_any
        return True

    def profit_target_progress_pct(self, current_equity: float, initial_equity: float) -> float:
        if initial_equity == 0:
            return 0.0
        return (current_equity - initial_equity) / initial_equity * 100.0

    def is_profit_soft_lock(self, progress_pct: float) -> bool:
        return progress_pct >= self.cfg.profit_target.soft_lock_pct

    def is_profit_hard_lock(self, progress_pct: float) -> bool:
        return progress_pct >= self.cfg.profit_target.hard_lock_pct

    def compute_stability_kpis(self, trade_pnls: List[float]) -> dict:
        if not trade_pnls:
            return {"profit_factor": 0.0, "winrate": 0.0, "pnl_std": 0.0, "mean_pnl": 0.0}
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
        winrate = len(wins) / len(trade_pnls)
        pnl_std = pstdev(trade_pnls) if len(trade_pnls) > 1 else 0.0
        mean_pnl = sum(trade_pnls) / len(trade_pnls)
        return {"profit_factor": profit_factor, "winrate": winrate, "pnl_std": pnl_std, "mean_pnl": mean_pnl}

    def is_stability_degraded(self, kpis: dict) -> bool:
        pf = kpis.get("profit_factor", 0.0)
        winrate = kpis.get("winrate", 0.0)
        pnl_std = kpis.get("pnl_std", 0.0)
        mean_pnl = kpis.get("mean_pnl", 0.0)
        if pf < self.cfg.stability.min_profit_factor:
            return True
        if winrate < self.cfg.stability.min_winrate:
            return True
        threshold = max(abs(mean_pnl), 1e-6) * self.cfg.stability.max_pnl_std_multiple
        return pnl_std > threshold

    def check_circuit_breaker(
        self,
        *,
        last_equity: float,
        current_equity: float,
        last_trade_slippage_pips: Optional[float],
        now: datetime,
    ) -> bool:
        instant_loss_pct = 0.0
        if last_equity > 0:
            instant_loss_pct = max(0.0, (last_equity - current_equity) / last_equity * 100.0)
        slippage_bad = (
            last_trade_slippage_pips is not None and last_trade_slippage_pips > self.cfg.circuit_breaker.max_slippage_pips
        )
        if instant_loss_pct > self.cfg.circuit_breaker.max_instant_loss_pct or slippage_bad:
            freeze_delta = timedelta(minutes=self.cfg.circuit_breaker.freeze_minutes)
            self.state.circuit_freeze_until = now + freeze_delta
            return True
        return False

    def is_circuit_breaker_active(self, now: datetime) -> bool:
        if self.state.circuit_freeze_until is None:
            return False
        return now <= self.state.circuit_freeze_until
