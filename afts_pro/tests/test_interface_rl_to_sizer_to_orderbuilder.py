import pandas as pd

from afts_pro.core.models import MarketState, StrategyDecision
from afts_pro.exec.order_builder import OrderBuilder
from afts_pro.exec.position_models import AccountState
from afts_pro.exec.position_sizer import PositionSizer, PositionSizerConfig
from afts_pro.rl.rl_inference import RLInferenceHook, ObservationBuilder
from afts_pro.rl.types import RLObsSpec, ActionSpec
from afts_pro.rl.risk_agent import RiskAgent, RiskAgentConfig
from afts_pro.rl.exit_agent import ExitAgent, ExitAgentConfig


class DummyRisk(RiskAgent):
    def __init__(self, val: float, spec: RLObsSpec):
        super().__init__(RiskAgentConfig(exploration_epsilon=0.0, min_risk_pct=0.1, max_risk_pct=2.0), spec, ActionSpec(action_type="continuous", shape=(1,)))
        self.val = val

    def act(self, obs, deterministic: bool = False) -> float:
        return self.val


def _market() -> MarketState:
    return MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=100, high=100, low=100, close=100, volume=0.0)


def test_risk_pct_changes_position_size_and_order():
    obs_spec = RLObsSpec(shape=(4,), dtype="float32", as_dict=False)
    builder = ObservationBuilder(obs_spec)
    risk_agent = DummyRisk(2.0, obs_spec)
    hook = RLInferenceHook(risk_agent=risk_agent, exit_agent=None, obs_builder=builder)
    acc = AccountState(balance=10000, equity=10000, realized_pnl=0, unrealized_pnl=0, fees_total=0)
    decision = StrategyDecision(action="entry", side="long", confidence=1.0, update={"sl_price": 99.0})
    actions = hook.compute_actions(_market(), acc, None, None)
    hook.apply_to_decision(decision, actions)

    sizer = PositionSizer(PositionSizerConfig())
    sizing = sizer.compute_position_size(
        symbol="ETH",
        side="long",
        entry_price=100.0,
        sl_price=99.0,
        equity=acc.equity,
        agent_risk_pct=decision.update.get("risk_pct"),
    )
    decision.update["position_size"] = sizing.size

    ob = OrderBuilder(use_position_sizer=True)
    orders = ob.build_entry_orders(decision, _market(), acc)
    assert sizing.size > 0
    assert orders and orders[0].qty == sizing.size


def test_missing_risk_pct_falls_back_to_fixed():
    acc = AccountState(balance=10000, equity=10000, realized_pnl=0, unrealized_pnl=0, fees_total=0)
    decision = StrategyDecision(action="entry", side="long", confidence=1.0, update={"sl_price": 99.0})
    sizer = PositionSizer(PositionSizerConfig(base_risk_mode="fixed", fixed_risk_pct=0.5))
    sizing = sizer.compute_position_size(
        symbol="ETH",
        side="long",
        entry_price=100.0,
        sl_price=99.0,
        equity=acc.equity,
        agent_risk_pct=None,
    )
    assert sizing.effective_risk_pct == 0.5
