import logging
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from afts_pro.config import (
    load_all_configs_into_global,
    load_global_config_from_profile,
    load_yaml,
)
from afts_pro.config.loader import get_file_mtimes
from afts_pro.config.profile_config import get_profile_include_paths
from afts_pro.core import MarketState, StrategyDecision
from afts_pro.core.mode_dispatcher import Mode
from afts_pro.data import MarketStateBuilder, ParquetFeed, ExtrasLoader
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
from afts_pro.exec.exit_policy import ExitPolicyApplier, ExitPolicyConfig
from afts_pro.exec.position_sizer import PositionSizer, PositionSizerConfig
from afts_pro.core.strategy_profile import load_strategy_profile
from afts_pro.core.strategy_orb import ORBStrategy
from afts_pro.risk import (
    RiskManager,
    create_risk_policy_from_config,
    load_risk_config,
)
from afts_pro.risk.ftmo_rules import FtmoRiskEngine, FtmoRiskConfig
from afts_pro.behaviour import BehaviourManager
from afts_pro.config.behaviour_config import create_guards
from afts_pro.core.rl_hook_integration import integrate_rl_inference
from afts_pro.rl.types import RLObsSpec
from afts_pro.runlogger import RunLogger
from afts_pro.runlogger.models import RunMeta
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
        logger.info("TRAIN mode selected. Use CLI-based TrainController entrypoint for now.")
    elif mode == Mode.LIVE:
        logger.info("LIVE mode stub - no implementation yet.")


async def _run_simulation() -> None:
    _sanity_check_exec_models()

    profile_path = _PROFILE_PATH
    profile_name = Path(profile_path).stem if profile_path else "default"
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
    sim_mode_cfg_path = PROJECT_ROOT / "configs" / "modes" / "sim.yaml"
    sim_mode_cfg = load_yaml(str(sim_mode_cfg_path)) if sim_mode_cfg_path.exists() else {}
    use_risk_agent = bool(sim_mode_cfg.get("use_risk_agent", False))
    use_exit_agent = bool(sim_mode_cfg.get("use_exit_agent", False))
    agent_paths = sim_mode_cfg.get("agent_paths", {})
    use_position_sizer_flag = bool(sim_mode_cfg.get("use_position_sizer", False))
    use_risk_agent_for_sizing = bool(sim_mode_cfg.get("use_risk_agent_for_sizing", False))
    position_sizer_cfg_path = sim_mode_cfg.get("position_sizer_config", "configs/exec/position_sizer.yaml")
    position_sizer_cfg = load_yaml(str(PROJECT_ROOT / position_sizer_cfg_path)) if position_sizer_cfg_path else {}
    exit_policy_cfg_path = PROJECT_ROOT / "configs" / "exec" / "exit_policy.yaml"
    exit_policy_cfg = load_yaml(str(exit_policy_cfg_path)) if exit_policy_cfg_path.exists() else {}

    risk_config_path = Path(global_config.risk.policy_path)
    if not risk_config_path.is_absolute():
        risk_config_path = PROJECT_ROOT / risk_config_path
    risk_config = load_risk_config(str(risk_config_path))
    risk_policy = create_risk_policy_from_config(str(risk_config_path))
    starting_balance = float(risk_config.get("initial_balance", 100000.0))
    ftmo_engine = None
    ftmo_cfg_path = sim_mode_cfg.get("risk", {}).get("ftmo_config_path") if isinstance(sim_mode_cfg, dict) else None
    use_ftmo = sim_mode_cfg.get("risk", {}).get("use_ftmo_risk", False) if isinstance(sim_mode_cfg, dict) else False
    if use_ftmo and ftmo_cfg_path:
        ftmo_cfg = FtmoRiskConfig(**load_yaml(str(PROJECT_ROOT / ftmo_cfg_path)))
        ftmo_engine = FtmoRiskEngine(ftmo_cfg)

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
    order_builder = OrderBuilder(asset_specs=asset_specs, use_position_sizer=sim_mode_cfg.get("use_position_sizer", False))
    position_manager = PositionManager()
    execution_cfg = global_config.execution
    fill_engine = _build_fill_engine(execution_cfg)
    price_validator = PriceValidator()
    risk_manager = RiskManager(risk_policy, ftmo_engine=ftmo_engine)
    run_logger: RunLogger | None = None
    if global_config.runlogger.enabled:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{global_config.environment.mode}_{profile_name}"
        run_meta = RunMeta(
            run_id=run_id,
            mode=global_config.environment.mode,
            profile_name=profile_name,
            started_at=datetime.now(timezone.utc),
            finished_at=None,
            symbol=symbol,
            timeframe="unknown",
        )
        run_logger = RunLogger(run_meta, global_config.runlogger, PROJECT_ROOT)
    else:
        logger.info("RUNLOGGER_DISABLED")
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
    profile_path_override = sim_mode_cfg.get("strategy_profile_path")
    if profile_path_override:
        profile_cfg = load_strategy_profile(str(PROJECT_ROOT / profile_path_override))
        strategies = [ORBStrategy(profile_cfg.orb, profile_cfg.session, symbol=profile_cfg.symbol or symbol)] if profile_cfg.orb else []
        bridge = StrategyBridge(strategies, asset_specs=asset_specs)
    else:
        strategies = _instantiate_strategies(symbol, global_config.strategy.enabled_strategies)
        bridge = StrategyBridge(strategies, asset_specs=asset_specs)
    extras_loader: ExtrasLoader | None = None
    extras_map = {}
    if global_config.extras.enabled:
        extras_loader = ExtrasLoader(global_config.extras)
        logger.info(
            "EXTRAS_LOADER_ENABLED | datasets=%s",
            [d.name for d in global_config.extras.get_enabled_datasets()],
        )
        extras_map = extras_loader.load_for_symbol(symbol)
        if extras_map:
            logger.info("EXTRAS_ATTACHED | symbol=%s | datasets=%s", symbol, list(extras_map.keys()))
        else:
            logger.info("EXTRAS_ENABLED_BUT_EMPTY | symbol=%s", symbol)
    else:
        logger.info("EXTRAS_LOADER_DISABLED")
    feature_engine: FeatureEngine | None = None
    if global_config.features.enabled:
        feature_engine = FeatureEngine(global_config.features)
        if extras_map:
            feature_engine.attach_extras(extras_map)
    else:
        logger.info("FEATURE_ENGINE_DISABLED")
    # RL inference hook (optional)
    raw_feature_count = len(global_config.features.raw_features) if global_config.features else 0
    obs_length = raw_feature_count + 5  # features + position/risk block
    obs_spec = RLObsSpec(shape=(obs_length,), dtype="float32", as_dict=False)
    rl_hook = integrate_rl_inference(
        use_risk_agent=use_risk_agent,
        use_exit_agent=use_exit_agent,
        risk_agent_path=agent_paths.get("risk_agent"),
        exit_agent_path=agent_paths.get("exit_agent"),
        obs_spec=obs_spec,
    )
    if rl_hook:
        logger.info("RL inference hook enabled (risk=%s, exit=%s)", use_risk_agent, use_exit_agent)
    else:
        logger.info("RL inference hook disabled.")
    exit_policy_applier: ExitPolicyApplier | None = None
    if sim_mode_cfg.get("use_exit_agent", False):
        cfg = ExitPolicyConfig(**exit_policy_cfg) if isinstance(exit_policy_cfg, dict) else ExitPolicyConfig()
        exit_policy_applier = ExitPolicyApplier(cfg)
    position_sizer: PositionSizer | None = None
    if use_position_sizer_flag:
        ps_cfg = PositionSizerConfig(**position_sizer_cfg) if isinstance(position_sizer_cfg, dict) else PositionSizerConfig()
        position_sizer = PositionSizer(ps_cfg)

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
            if run_logger is not None and event.event_type == "CLOSED":
                run_logger.on_trade_close(event, ts=state.timestamp)
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
            if run_logger is not None:
                run_logger.finalize_and_persist(global_config.model_dump())
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
                if run_logger is not None:
                    run_logger.finalize_and_persist(global_config.model_dump())
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
        if rl_hook is not None:
            try:
                pos_state = account_state.positions.get(state.symbol)
                actions = rl_hook.compute_actions(state, account_state, pos_state, feature_bundle)
                rl_hook.apply_to_decision(decision, actions)
                logger.info(
                    "RL inference applied | risk_pct=%s | exit_action=%s",
                    actions.get("risk_pct"),
                    actions.get("exit_action"),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("RL inference failed: %s", exc)
        if exit_policy_applier is not None and decision.meta.get("exit_action") is not None:
            pos_state = account_state.positions.get(state.symbol)
            exit_policy_applier.apply(decision.meta.get("exit_action"), pos_state, state, decision, atr=None)
        # Position sizing for entries
        if position_sizer is not None and decision.action == "entry":
            agent_risk = decision.update.get("risk_pct") or decision.meta.get("risk_pct")
            if not use_risk_agent_for_sizing:
                agent_risk = None
            sl_price = decision.update.get("sl_price") or decision.update.get("new_sl")
            atr_val = feature_bundle.model.values[0] if feature_bundle and feature_bundle.model else None
            sizing_result = position_sizer.compute_position_size(
                symbol=state.symbol,
                side=decision.side or "long",
                entry_price=state.close,
                sl_price=sl_price,
                equity=account_state.equity,
                agent_risk_pct=agent_risk,
                daily_realized_pnl=account_state.realized_pnl,
                atr=atr_val,
            )
            decision.update["position_size"] = sizing_result.size
            decision.meta["effective_risk_pct"] = sizing_result.effective_risk_pct
            decision.meta["risk_capped_by"] = sizing_result.capped_by
            logger.info(
                "Position size computed | size=%.4f | eff_risk=%.3f%% | caps=%s",
                sizing_result.size,
                sizing_result.effective_risk_pct,
                sizing_result.capped_by,
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
        if run_logger is not None:
            run_logger.on_bar_equity_snapshot(state.timestamp, account_state, risk_meta=risk_decision.meta)

        last_bar = state
        bar_index += 1

    if run_logger is not None:
        run_logger.finalize_and_persist(global_config.model_dump())


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
