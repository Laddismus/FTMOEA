import numpy as np

from afts_pro.core.models import MarketState, OHLCV
from afts_pro.exec.position_models import AccountState, Position, PositionSide
from afts_pro.features.state import RawFeatureState, FeatureBundle
from afts_pro.rl.env_features import EnvFeatureConfig, FtmoFeatureConfig
from afts_pro.rl.rl_inference import ObservationBuilder
from afts_pro.rl.types import RLObsSpec


class DummyAccount(AccountState):
    def __init__(self):
        super().__init__(
            balance=0.0,
            equity=100000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            fees_total=0.0,
            positions={},
            open_orders={},
        )
        self.current_spread_pips = 0.5


def _builder(expected_len: int) -> ObservationBuilder:
    feat_cfg = EnvFeatureConfig(ftmo=FtmoFeatureConfig(sessions=["London", "NewYork"]))
    return ObservationBuilder(obs_spec=RLObsSpec(shape=(expected_len,)), feature_config=feat_cfg)


def test_obs_includes_ftmo_features_when_enabled():
    # Raw feature bundle with two features
    raw = RawFeatureState(values={"a": 1.0, "b": 2.0})
    fb = FeatureBundle(raw=raw, model=None, extras=None)
    ms = MarketState(timestamp=None, symbol="SYM", ohlcv=OHLCV(open=1, high=2, low=0.5, close=1.5, volume=10), extras={})
    pos = Position(symbol="SYM", side=PositionSide.LONG, qty=1.0, entry_price=1.0, realized_pnl=0.0, unrealized_pnl=0.2, avg_entry_fees=0.0)
    acc = DummyAccount()
    meta = {
        "ftmo_daily_loss_pct": 2.0,
        "ftmo_overall_loss_pct": 4.0,
        "ftmo_plus_stage": 1,
        "ftmo_plus_rolling_loss_pct": 1.2,
        "ftmo_plus_loss_velocity": 5.0,
        "ftmo_plus_profit_progress_pct": 60.0,
        "ftmo_plus_session_name": "London",
        "ftmo_plus_pf": 1.5,
        "ftmo_plus_winrate": 0.55,
        "ftmo_plus_pnl_std": 0.8,
        "ftmo_plus_blocked_news": True,
    }
    expected_len = 26
    builder = _builder(expected_len)
    obs = builder.build(ms, pos, acc, meta, fb)
    assert len(obs) == expected_len
    # News flag should be 1.0
    assert obs.tolist().count(1.0) >= 1
    assert np.all(np.isfinite(obs))
    assert np.all(obs <= 1.0 + 1e-6)
    assert np.all(obs >= -1.0 - 1e-6)


def test_obs_omits_ftmo_features_when_disabled():
    raw = RawFeatureState(values={"a": 1.0})
    fb = FeatureBundle(raw=raw, model=None, extras=None)
    ms = MarketState(timestamp=None, symbol="SYM", ohlcv=OHLCV(open=1, high=1, low=1, close=1, volume=1), extras={})
    pos = None
    acc = DummyAccount()
    feat_cfg = EnvFeatureConfig(base_price_features=True, base_pnl_features=True, ftmo=FtmoFeatureConfig(include_daily_dd_pct=False, include_overall_dd_pct=False, include_stage=False, include_stage_one_hot=False, include_rolling_dd_pct=False, include_loss_velocity=False, include_profit_progress_pct=False, include_session_one_hot=False, include_news_flag=False, include_time_fence_flag=False, include_spread=False, include_stability_kpis=False, include_circuit_active_flag=False, sessions=[]))
    builder = ObservationBuilder(obs_spec=RLObsSpec(shape=(6,)), feature_config=feat_cfg)
    obs = builder.build(ms, pos, acc, {}, fb)
    # Only raw feature (1) + position (3) + pnl (2) = 6
    assert len(obs) == 6
    assert np.all(np.isfinite(obs))


def test_obs_deterministic_given_same_input():
    raw = RawFeatureState(values={"x": 1.0})
    fb = FeatureBundle(raw=raw, model=None, extras=None)
    ms = MarketState(timestamp=None, symbol="SYM", ohlcv=OHLCV(open=1, high=1, low=1, close=1, volume=1), extras={})
    pos = None
    acc = DummyAccount()
    meta = {"ftmo_daily_loss_pct": 1.0, "ftmo_plus_stage": 0}
    builder = _builder(10)
    obs1 = builder.build(ms, pos, acc, meta, fb)
    obs2 = builder.build(ms, pos, acc, meta, fb)
    assert np.array_equal(obs1, obs2)
