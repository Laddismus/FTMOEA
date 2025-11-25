import logging
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

from afts_pro.config import (
    load_all_configs_into_global,
    load_global_config_from_profile,
)
from afts_pro.config.loader import get_file_mtimes
from afts_pro.config.profile_config import get_profile_include_paths
from afts_pro.core import MarketState, StrategyDecision
from afts_pro.core.mode_dispatcher import Mode
from afts_pro.data import MarketStateBuilder, ParquetFeed
from afts_pro.exec import (
    AccountState,
    Fill,
    Order,
    OrderBuilder,
    OrderSide,
    OrderType,
    PositionEvent,
    PositionManager,
    PositionSide,
    SimFillEngine,
)
from afts_pro.risk import (
    RiskManager,
    create_risk_policy_from_config,
    load_risk_config,
)
from afts_pro.behaviour import BehaviourManager
from afts_pro.config.behaviour_config import create_guards
from afts_pro.sim.price_validator import PriceValidator
from afts_pro.strategies import DummyMLStrategy, OrbStrategy, StrategyBridge, StrategyRegistry
from afts_pro.features import FeatureEngine

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"


_PROFILE_PATH: Optional[str] = None


def set_profile_path(profile_path: Optional[str]) -> None:
    global _PROFILE_PATH
    _PROFILE_PATH = profile_path


def _build_fill_engine(execution_cfg) -> SimFillEngine:
    return SimFillEngine(
        fee_rate=execution_cfg.taker_fee_pct,
        slippage_ticks=0.0,
        tick_size=execution_cfg.tick_size,
        slippage_pct=execution_cfg.max_slippage_pct,
    )


def _instantiate_strategies(symbol: str, enabled: Sequence[str]) -> List:
    strategies = []
    for name in enabled:
        strategy_cls = StrategyRegistry.get(name)
        if strategy_cls is None:
            logger.warning("Strategy %s not registered; skipping.", name)
            continue
        strategies.append(strategy_cls(symbol=symbol))  # type: ignore[call-arg]
    if not strategies:
        logger.info("No strategies enabled; defaulting to ORB.")
        strategies = [OrbStrategy(symbol=symbol)]
    return strategies


def _collect_config_paths(profile_path: Optional[str], global_config) -> List[Path]:
    paths: List[Path] = []
    if profile_path:
        paths.extend(get_profile_include_paths(profile_path))
    if global_config.source_paths:
        for p in global_config.source_paths.values():
            paths.append(Path(p))
    paths.append(Path(global_config.risk.policy_path))
    # de-duplicate
    unique: Dict[Path, None] = {}
    for p in paths:
        unique[p.resolve()] = None
    return list(unique.keys())


async def start(mode: Mode, profile_path: Optional[str] = None) -> None:
    """
    Primary asynchronous entrypoint for the trading engine.
    """
    if profile_path is not None:
        set_profile_path(profile_path)
    logger.info("Engine start invoked for mode=%s", mode.value)

    if mode == Mode.SIM:
        await _run_simulation()
    elif mode == Mode.TRAIN:
        logger.info("TRAIN mode stub - no implementation yet.")
    elif mode == Mode.LIVE:
        logger.info("LIVE mode stub - no implementation yet.")


async def _run_simulation() -> None:
    _sanity_check_exec_models()

    profile_path = _PROFILE_PATH
    if profile_path is not None:
        global_config = load_global_config_from_profile(profile_path)
        logger.info("PROFILE_SELECTED | name=%s | path=%s", Path(profile_path).stem, profile_path)
    else:
        global_config = load_all_configs_into_global()
        logger.info("Loaded GlobalConfig: %s", global_config.summary())

    feed = ParquetFeed(DATA_ROOT)
    builder = MarketStateBuilder(feed)
    asset_specs = global_config.assets.assets
    symbol = next(iter(asset_specs.keys()), "ETHUSDT_5T")

    risk_config_path = Path(global_config.risk.policy_path)
    if not risk_config_path.is_absolute():
        risk_config_path = PROJECT_ROOT / risk_config_path
    risk_config = load_risk_config(str(risk_config_path))
    risk_policy = create_risk_policy_from_config(str(risk_config_path))
    starting_balance = float(risk_config.get("initial_balance", 100000.0))

    logger.info(
        "RISK_CONFIG | path=%s | type=%s | params=%s",
        risk_config_path,
        risk_config.get("type"),
        {k: v for k, v in risk_config.items() if k != "type"},
    )

    account_state = AccountState(
        balance=starting_balance,
        equity=starting_balance,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        fees_total=0.0,
    )
    order_builder = OrderBuilder(asset_specs=asset_specs)
    position_manager = PositionManager()
    execution_cfg = global_config.execution
    fill_engine = _build_fill_engine(execution_cfg)
    price_validator = PriceValidator()
    risk_manager = RiskManager(risk_policy)
    behaviour_manager: BehaviourManager | None = None
    behaviour_guards = create_guards(global_config.behaviour, initial_balance=starting_balance)
    if behaviour_guards:
        logger.info(
            "BEHAVIOUR_CONFIG | enabled_guards=%s",
            [g.name for g in behaviour_guards],
        )
        behaviour_manager = BehaviourManager(guards=behaviour_guards)
    else:
        logger.info("Behaviour layer disabled or no guards configured.")
    strategies = _instantiate_strategies(symbol, global_config.strategy.enabled_strategies)
    bridge = StrategyBridge(strategies, asset_specs=asset_specs)
    feature_engine: FeatureEngine | None = None
    if global_config.features.enabled:
        feature_engine = FeatureEngine(global_config.features)
    else:
        logger.info("FEATURE_ENGINE_DISABLED")

    tracked_paths: List[Path] = []
    config_mtimes: Dict[Path, float] = {}
    if global_config.environment.config_hot_reload_enabled and global_config.environment.mode in {"sim", "train"}:
        tracked_paths = _collect_config_paths(profile_path, global_config)
        config_mtimes = get_file_mtimes(tracked_paths)

    last_bar: Optional[MarketState] = None
    pending_orders_for_next_bar: List[Order] = []
    demo_entry_sent = False
    market_states = builder.iter_market_states(symbol=symbol, folder="final_agg")
    bar_index = 0
    for state in islice(market_states, 200):
        price_validator.validate_bar_sequence(last_bar, state)
        logger.info(
            "SIM bar | ts=%s | symbol=%s | close=%.4f",
            state.timestamp.isoformat(),
            state.symbol,
            state.close,
        )
        (
            global_config,
            behaviour_manager,
            strategies,
            fill_engine,
            config_mtimes,
            tracked_paths,
            bridge,
            execution_cfg,
            asset_specs,
        ) = maybe_reload_config(
            bar_index=bar_index,
            profile_path=profile_path,
            global_config=global_config,
            current_behaviour_manager=behaviour_manager,
            current_strategies=strategies,
            current_execution_engine=fill_engine,
            current_bridge=bridge,
            symbol=symbol,
            tracked_paths=tracked_paths,
            last_mtimes=config_mtimes,
            account_state=account_state,
        )
        order_builder.asset_specs = asset_specs

        # Activate pending orders only for this bar (generated previous loop)
        if pending_orders_for_next_bar:
            for order in pending_orders_for_next_bar:
                order.created_at = order.created_at or (last_bar.timestamp if last_bar else state.timestamp)
                account_state.open_orders[order.id] = order
            logger.debug("Activated pending orders: %s", [o.id for o in pending_orders_for_next_bar])
            pending_orders_for_next_bar.clear()

        # Fill existing open orders using current bar; requires at least one prior bar
        if last_bar is not None:
            fills = fill_engine.process_bar(
                account_state=account_state,
                open_orders=account_state.open_orders,
                market_state=state,
                last_bar=last_bar,
            )
        else:
            fills = []

        had_position = state.symbol in account_state.positions
        for fill in fills:
            order_ref = account_state.open_orders.get(fill.order_id)
            if order_ref:
                try:
                    price_validator.validate_fill_timing(order_ref, fill.timestamp)
                except ValueError as exc:
                    logger.error("Fill timing violated: %s", exc)
                    continue

            logger.debug("Applying fill: %s", fill)
            try:
                event = position_manager.apply_fill(fill, account_state)
            except ValueError as exc:
                logger.warning("Fill application blocked: %s", exc)
                continue
            if (
                behaviour_manager is not None
                and event.event_type in {"CLOSED", "REDUCED"}
                and event.realized_pnl_delta != 0.0
            ):
                behaviour_manager.on_trade_closed(
                    trade_pnl=event.realized_pnl_delta,
                    ts=state.timestamp,
                    account_state=account_state,
                )
            reason = "SL" if fill.meta.get("is_sl") else "TP" if fill.meta.get("is_tp") else "FILL"
            logger.info(
                "FILL | ts=%s | symbol=%s | side=%s | qty=%.4f | price=%.4f | reason=%s | fee=%.4f",
                fill.timestamp.isoformat(),
                fill.symbol,
                fill.side.value,
                fill.qty,
                fill.price,
                reason,
                fill.fee,
            )

        position_manager.update_unrealized_pnl(account_state, market_price=state.close)

        risk_decision = risk_manager.before_new_orders(account_state, state.timestamp)
        logger.info(
            "RISK | ts=%s | policy=%s | allow_new_orders=%s | hard_stop=%s | reason=%s",
            state.timestamp.isoformat(),
            risk_policy.name,
            risk_decision.allow_new_orders,
            risk_decision.hard_stop_trading,
            risk_decision.reason,
        )
        logger.debug("RISK_META | %s", risk_decision.meta)
        if risk_decision.hard_stop_trading:
            logger.info("RISK HARD STOP | trading halted by FTMO policy")
            return

        behaviour_decision = None
        if risk_decision.allow_new_orders and behaviour_manager is not None:
            behaviour_decision = behaviour_manager.before_new_orders(ts=state.timestamp, account_state=account_state)
            meta_short = behaviour_decision.meta if behaviour_decision.meta else {}
            logger.info(
                "BEHAVIOUR | ts=%s | allow_new_orders=%s | hard_block=%s | reason=%s | meta_short=%s",
                state.timestamp.isoformat(),
                behaviour_decision.allow_new_orders,
                behaviour_decision.hard_block_trading,
                behaviour_decision.reason,
                meta_short,
            )
            if behaviour_decision.hard_block_trading:
                logger.info("BEHAVIOUR HARD BLOCK | trading halted by guards")
                return
        feature_bundle = feature_engine.update(state) if feature_engine is not None else None
        if feature_bundle and logger.isEnabledFor(logging.DEBUG) and bar_index < 5:
            logger.debug(
                "FEATURES | ts=%s | raw_keys=%s | model_len=%d",
                state.timestamp.isoformat(),
                list(feature_bundle.raw.values.keys()),
                len(feature_bundle.model.values) if feature_bundle.model else 0,
            )

        if behaviour_decision is not None and not behaviour_decision.allow_new_orders:
            decision = StrategyDecision(action="none", side=None, confidence=0.0)
            new_orders = []
        else:
            decision = bridge.on_bar(state, features=feature_bundle)
            meta_short = decision.meta.get("strategies", decision.meta)
            logger.info(
                "DECISION | ts=%s | action=%s | side=%s | conf=%.3f | meta_short=%s",
                state.timestamp.isoformat(),
                decision.action,
                decision.side,
                decision.confidence,
                meta_short,
            )

            new_orders: List[Order] = []
            if risk_decision.allow_new_orders:
                new_orders.extend(order_builder.build_entry_orders(decision, state, account_state))
                new_orders.extend(order_builder.build_manage_orders(decision, state, account_state))
                new_orders.extend(order_builder.build_exit_orders(decision, state, account_state))

                if not new_orders and not account_state.open_orders and not account_state.positions and not demo_entry_sent:
                    demo_decision = StrategyDecision(action="entry", side="long", confidence=1.0)
                    new_orders.extend(order_builder.build_entry_orders(demo_decision, state, account_state))
                    demo_entry_sent = True
            else:
                logger.debug("Risk blocked new orders this bar.")

        if new_orders:
            pending_orders_for_next_bar.extend(new_orders)
            logger.info(
                "ORDERS_BUILT | count=%d | reduce_only_flags=%s",
                len(new_orders),
                [o.reduce_only for o in new_orders],
            )
            logger.debug("Built orders (pending for next bar): %s", new_orders)
        else:
            logger.debug("ORDERS_BUILT | count=0")

        position = account_state.positions.get(state.symbol)
        if position:
            if not had_position:
                logger.info(
                    "POSITION_OPENED | ts=%s | symbol=%s | side=%s | qty=%.4f | entry=%.4f | realized=%.4f",
                    state.timestamp.isoformat(),
                    position.symbol,
                    position.side.value,
                    position.qty,
                    position.entry_price,
                    position.realized_pnl,
                )
            else:
                logger.info(
                    "POSITION | symbol=%s | qty=%.4f | entry=%.4f | realized=%.4f | unrealized=%.4f",
                    position.symbol,
                    position.qty,
                    position.entry_price,
                    position.realized_pnl,
                    position.unrealized_pnl,
                )
        else:
            if had_position:
                logger.info(
                    "POSITION_CLOSED | ts=%s | symbol=%s | realized=%.4f",
                    state.timestamp.isoformat(),
                    state.symbol,
                    account_state.realized_pnl,
                )
            else:
                logger.info("POSITION | symbol=%s | no open position", state.symbol)

        last_bar = state
        bar_index += 1


def _sanity_check_exec_models() -> None:
    """
    Ensure exec-layer models can be instantiated (import smoke test).
    """
    _ = Order(
        id="dummy-order",
        client_order_id="client-1",
        symbol="ETHUSDT",
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        qty=1.0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    _ = AccountState(
        balance=0.0,
        equity=0.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        fees_total=0.0,
    )


def maybe_reload_config(
    *,
    bar_index: int,
    profile_path: Optional[str],
    global_config,
    current_behaviour_manager: Optional[BehaviourManager],
    current_strategies: List,
    current_execution_engine: SimFillEngine,
    current_bridge: StrategyBridge,
    symbol: str,
    tracked_paths: List[Path],
    last_mtimes: Dict[Path, float],
    account_state: AccountState,
) -> Tuple:
    env_cfg = global_config.environment
    if not env_cfg.config_hot_reload_enabled or env_cfg.mode not in {"sim", "train"}:
        return (
            global_config,
            current_behaviour_manager,
            current_strategies,
            current_execution_engine,
            last_mtimes,
            tracked_paths,
            current_bridge,
            global_config.execution,
            global_config.assets.assets,
        )

    interval = max(env_cfg.config_hot_reload_interval_bars, 0)
    if interval == 0 or bar_index % interval != 0:
        return (
            global_config,
            current_behaviour_manager,
            current_strategies,
            current_execution_engine,
            last_mtimes,
            tracked_paths,
            current_bridge,
            global_config.execution,
            global_config.assets.assets,
        )

    if not tracked_paths:
        tracked_paths = _collect_config_paths(profile_path, global_config)
    new_mtimes = get_file_mtimes(tracked_paths)
    changed_files = [str(p) for p in tracked_paths if new_mtimes.get(p) != last_mtimes.get(p)]
    if not changed_files:
        return (
            global_config,
            current_behaviour_manager,
            current_strategies,
            current_execution_engine,
            new_mtimes,
            tracked_paths,
            current_bridge,
            global_config.execution,
            global_config.assets.assets,
        )

    logger.info("CONFIG_HOT_RELOAD_TRIGGERED | changed_files=%s", changed_files)

    if profile_path:
        new_global_config = load_global_config_from_profile(profile_path)
    else:
        new_global_config = load_all_configs_into_global()

    scope = set(env_cfg.config_hot_reload_scope or [])

    new_behaviour_manager = current_behaviour_manager
    if "behaviour" in scope:
        behaviour_guards = create_guards(
            new_global_config.behaviour,
            initial_balance=float(account_state.balance),
        )
        if behaviour_guards:
            new_behaviour_manager = BehaviourManager(guards=behaviour_guards)
        else:
            new_behaviour_manager = None

    new_strategies = current_strategies
    if "strategy" in scope:
        new_strategies = _instantiate_strategies(symbol, new_global_config.strategy.enabled_strategies)

    new_execution_engine = current_execution_engine
    new_execution_cfg = new_global_config.execution
    if "execution" in scope:
        new_execution_engine = _build_fill_engine(new_execution_cfg)

    if new_global_config.risk.policy_path != global_config.risk.policy_path or (
        new_global_config.risk.policy_type != global_config.risk.policy_type
    ):
        logger.info("CONFIG_HOT_RELOAD_RISK_CHANGE_IGNORED_IN_V1 | from=%s to=%s", global_config.risk.policy_path, new_global_config.risk.policy_path)

    new_asset_specs = new_global_config.assets.assets
    new_bridge = current_bridge
    if new_strategies is not current_strategies:
        new_bridge = StrategyBridge(new_strategies, asset_specs=new_asset_specs)

    tracked_paths = _collect_config_paths(profile_path, new_global_config)
    new_mtimes = get_file_mtimes(tracked_paths)

    return (
        new_global_config,
        new_behaviour_manager,
        new_strategies,
        new_execution_engine,
        new_mtimes,
        tracked_paths,
        new_bridge,
        new_execution_cfg,
        new_asset_specs,
    )
