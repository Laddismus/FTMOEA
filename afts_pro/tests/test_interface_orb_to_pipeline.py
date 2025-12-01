import pandas as pd

from afts_pro.core.strategy_profile import load_strategy_profile
from afts_pro.core.strategy_orb import ORBStrategy
from afts_pro.strategies.bridge import StrategyBridge
from afts_pro.core.models import StrategyDecision, MarketState
from afts_pro.exec.position_models import AccountState
from afts_pro.rl.rl_inference import RLInferenceHook, ObservationBuilder
from afts_pro.rl.types import RLObsSpec, ActionSpec
from afts_pro.rl.risk_agent import RiskAgent, RiskAgentConfig
from afts_pro.exec.position_sizer import PositionSizer, PositionSizerConfig
from afts_pro.exec.order_builder import OrderBuilder


class DummyRisk(RiskAgent):
    def __init__(self, val: float, spec: RLObsSpec):
        super().__init__(RiskAgentConfig(exploration_epsilon=0.0, min_risk_pct=0.1, max_risk_pct=2.0), spec, ActionSpec(action_type="continuous", shape=(1,)))
        self.val = val

    def act(self, obs, deterministic: bool = False) -> float:
        return self.val


def _market(ts: str, high: float, low: float, close: float) -> MarketState:
    return MarketState(timestamp=pd.Timestamp(ts), symbol="EURUSD", open=close, high=high, low=low, close=close, volume=0.0)


def test_orb_breakout_flows_to_order_builder(tmp_path):
    profile = load_strategy_profile("configs/strategy/orb_15m_v1.yaml")
    strategy = ORBStrategy(profile.orb, profile.session, symbol=profile.symbol)
    bridge = StrategyBridge([strategy], asset_specs={})
    acc = AccountState(balance=10000, equity=10000, realized_pnl=0, unrealized_pnl=0, fees_total=0)
    obs_spec = RLObsSpec(shape=(4,), dtype="float32", as_dict=False)
    builder = ObservationBuilder(obs_spec)
    risk_agent = DummyRisk(1.0, obs_spec)
    rl_hook = RLInferenceHook(risk_agent=risk_agent, exit_agent=None, obs_builder=builder)
    sizer = PositionSizer(PositionSizerConfig())
    ob = OrderBuilder(use_position_sizer=True)

    bar1 = _market("2024-01-01 08:00", high=1.1010, low=1.1000, close=1.1005)
    decision1 = bridge.on_bar(bar1, features=None)
    assert decision1.action == "none"

    bar2 = _market("2024-01-01 08:16", high=1.1013, low=1.1005, close=1.1013)
    decision2 = bridge.on_bar(bar2, features=None)
    actions = rl_hook.compute_actions(bar2, acc, None, None)
    rl_hook.apply_to_decision(decision2, actions)
    sizing = sizer.compute_position_size(
        symbol=bar2.symbol,
        side=decision2.side or "long",
        entry_price=bar2.close,
        sl_price=decision2.update.get("sl_price"),
        equity=acc.equity,
        agent_risk_pct=decision2.update.get("risk_pct"),
    )
    decision2.update["position_size"] = sizing.size
    orders = ob.build_entry_orders(decision2, bar2, acc)
    assert decision2.action == "entry"
    assert orders and orders[0].qty == sizing.size
