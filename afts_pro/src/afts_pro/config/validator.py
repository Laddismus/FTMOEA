from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from afts_pro.config.global_config import GlobalConfig
from afts_pro.strategies import StrategyRegistry

logger = logging.getLogger(__name__)


def _as_error(message: str) -> str:
    return f"ERROR: {message}"


def _as_warn(message: str) -> str:
    return f"WARN: {message}"


def validate_paths(global_config: GlobalConfig) -> List[str]:
    messages: List[str] = []
    source_paths = global_config.source_paths or {}

    # Risk policy path
    risk_path = Path(global_config.risk.policy_path)
    if not risk_path.exists():
        messages.append(_as_error(f"Risk policy path missing: {risk_path}"))

    # Included config files if available
    for label, path_str in source_paths.items():
        path_obj = Path(path_str)
        if not path_obj.exists():
            messages.append(_as_warn(f"Config include missing ({label}): {path_obj}"))
    return messages


def validate_assets(global_config: GlobalConfig, data_root: Path) -> List[str]:
    messages: List[str] = []
    data_root = data_root.resolve()
    for symbol in global_config.assets.assets.keys():
        pattern = f"{symbol}*.parquet"
        matches = list((data_root).glob(pattern))
        if not matches:
            messages.append(_as_warn(f"No parquet files found for asset symbol={symbol} under {data_root}"))
    return messages


def validate_strategies(global_config: GlobalConfig) -> List[str]:
    messages: List[str] = []
    available = StrategyRegistry.available()
    enabled = global_config.strategy.enabled_strategies

    for strategy_name in enabled:
        if strategy_name not in available:
            messages.append(_as_error(f"Unknown strategy enabled: {strategy_name}"))

    for param_name in global_config.strategy.strategy_params.keys():
        if param_name not in available:
            messages.append(_as_warn(f"Params provided for unknown strategy: {param_name}"))

    return messages


def validate_behaviour(global_config: GlobalConfig) -> List[str]:
    messages: List[str] = []
    if not global_config.behaviour.enabled:
        return messages
    guards = getattr(global_config.behaviour, "guards", None)
    if guards is None:
        messages.append(_as_error("Behaviour enabled but no guards configuration present"))
        return messages

    def _check_positive(name: str, value: float | int | None) -> None:
        if value is not None and value < 0:
            messages.append(_as_error(f"{name} must be non-negative (got {value})"))

    for guard in [
        ("max_trades_per_day", getattr(guards, "max_trades_per_day", None), "max_trades_per_day"),
        ("max_consecutive_losses", getattr(guards, "max_consecutive_losses", None), "max_consecutive_losses"),
        ("cooldown_after_loss", getattr(guards, "cooldown_after_loss", None), "cooldown_minutes"),
        ("daily_pnl", getattr(guards, "daily_pnl", None), "max_daily_loss_pct_initial"),
        ("big_loss_cooldown", getattr(guards, "big_loss_cooldown", None), "big_loss_pct_initial"),
    ]:
        guard_name, guard_cfg, field = guard
        if guard_cfg and getattr(guard_cfg, "enabled", False):
            params = guard_cfg.params
            value = params.get(field)
            _check_positive(f"{guard_name}.{field}", value)
    return messages


def validate_features(global_config: GlobalConfig) -> List[str]:
    messages: List[str] = []
    cfg = getattr(global_config, "features", None)
    if cfg is None:
        return messages
    if not cfg.enabled:
        return messages

    names = [rf.name for rf in cfg.raw_features]
    if len(names) != len(set(names)):
        messages.append(_as_error("Duplicate raw feature names detected."))

    valid_calcs = {
        "close_return",
        "rolling_vol",
        "atr",
        "ema",
        "rsi",
        "volatility_score",
        "trend_score",
    }
    for rf in cfg.raw_features:
        if rf.calculator not in valid_calcs:
            messages.append(_as_error(f"Unknown feature calculator '{rf.calculator}' in feature '{rf.name}'"))

    if cfg.model_features.enabled:
        for name in cfg.model_features.feature_order:
            if name not in names:
                messages.append(_as_error(f"Model feature order references unknown raw feature: {name}"))
        scaling = cfg.model_features.scaling
        if scaling.type == "zscore":
            params = scaling.params.get("zscore", {})
            means = params.get("means", {})
            stds = params.get("stds", {})
            for fname in cfg.model_features.feature_order:
                if fname not in means or fname not in stds:
                    messages.append(_as_error(f"Missing zscore params for feature '{fname}'"))
                else:
                    if stds.get(fname, 0) <= 0:
                        messages.append(_as_warn(f"Non-positive std for feature '{fname}' in zscore scaling"))
        elif scaling.type == "minmax":
            params = scaling.params.get("minmax", {})
            mins = params.get("mins", {})
            maxs = params.get("maxs", {})
            for fname in cfg.model_features.feature_order:
                if fname not in mins or fname not in maxs:
                    messages.append(_as_error(f"Missing minmax params for feature '{fname}'"))
                else:
                    if maxs.get(fname, 0.0) <= mins.get(fname, 0.0):
                        messages.append(_as_error(f"Invalid minmax range for feature '{fname}' (max <= min)"))
    if cfg.enabled and not cfg.raw_features:
        messages.append(_as_warn("Features enabled but no raw_features defined."))
    return messages


def run_all_validations(global_config: GlobalConfig, data_root: Path) -> Tuple[bool, List[str]]:
    messages: List[str] = []
    messages.extend(validate_paths(global_config))
    messages.extend(validate_assets(global_config, data_root))
    messages.extend(validate_strategies(global_config))
    messages.extend(validate_behaviour(global_config))
    messages.extend(validate_features(global_config))

    has_error = any(msg.startswith("ERROR") for msg in messages)
    return (not has_error, messages)
