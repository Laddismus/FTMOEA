from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from afts_pro.config.asset_config import AssetConfig
from afts_pro.config.behaviour_config import BehaviourConfig, create_guards, load_behaviour_config
from afts_pro.config.base_models import BaseConfigModel
from afts_pro.config.environment_config import EnvironmentConfig
from afts_pro.config.execution_config import ExecutionConfig
from afts_pro.config.feature_config import FeatureConfig, load_feature_config
from afts_pro.config.extras_config import ExtrasConfig, load_extras_config
from afts_pro.config.loader import load_yaml
from afts_pro.config.profile_config import ProfileConfig, load_profile
from afts_pro.config.runlogger_config import RunLoggerConfig, load_runlogger_config
from afts_pro.config.risk_config import RiskConfig
from afts_pro.config.strategy_config import StrategyConfig

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "configs"


class GlobalConfig(BaseConfigModel):
    environment: EnvironmentConfig
    execution: ExecutionConfig
    assets: AssetConfig
    risk: RiskConfig
    behaviour: BehaviourConfig
    strategy: StrategyConfig
    source_paths: Dict[str, str] | None = None
    features: FeatureConfig
    extras: ExtrasConfig
    runlogger: RunLoggerConfig

    def summary(self) -> str:
        return (
            f"mode={self.environment.mode} | "
            f"risk={self.risk.policy_type} | "
            f"strategies={self.strategy.enabled_strategies} | "
            f"behaviour_enabled={self.behaviour.enabled} | "
            f"assets={list(self.assets.assets.keys())}"
        )


def _load_environment_config(path: Path) -> EnvironmentConfig:
    data = load_yaml(str(path)).get("environment", {})
    return EnvironmentConfig(**data)


def _load_execution_config(path: Path) -> ExecutionConfig:
    data = load_yaml(str(path)).get("execution", {})
    return ExecutionConfig(**data)


def _load_asset_config(path: Path) -> AssetConfig:
    data = load_yaml(str(path)).get("assets", {})
    return AssetConfig(assets=data)


def _load_strategy_config(path: Path) -> StrategyConfig:
    data = load_yaml(str(path)).get("strategy", {})
    return StrategyConfig(**data)


def _load_risk_config(path: Path) -> RiskConfig:
    data = load_yaml(str(path)).get("risk", {})
    return RiskConfig(**data)


def _load_behaviour_config(path: Path) -> BehaviourConfig:
    return load_behaviour_config(str(path))


def load_global_config(
    *,
    environment_path: str = str(CONFIG_ROOT / "environment.yaml"),
) -> GlobalConfig:
    env_cfg = _load_environment_config(Path(environment_path))

    execution_path = CONFIG_ROOT / "execution.yaml"
    assets_path = CONFIG_ROOT / "assets.yaml"
    strategy_path = CONFIG_ROOT / "strategy.yaml"
    risk_wrapper_path = CONFIG_ROOT / "risk" / "risk.yaml"
    behaviour_path = CONFIG_ROOT / "behaviour" / "default.yaml"
    features_path = CONFIG_ROOT / "features.yaml"
    extras_path = CONFIG_ROOT / "extras.yaml"
    runlogger_path = CONFIG_ROOT / "runlogger.yaml"

    execution_cfg = _load_execution_config(execution_path)
    assets_cfg = _load_asset_config(assets_path)
    strategy_cfg = _load_strategy_config(strategy_path)
    risk_cfg = _load_risk_config(risk_wrapper_path)
    behaviour_cfg = _load_behaviour_config(behaviour_path)
    feature_cfg = load_feature_config(str(features_path))
    extras_cfg = load_extras_config(str(extras_path))
    runlogger_cfg = load_runlogger_config(str(runlogger_path))

    global_cfg = GlobalConfig(
        environment=env_cfg,
        execution=execution_cfg,
        assets=assets_cfg,
        risk=risk_cfg,
        behaviour=behaviour_cfg,
        strategy=strategy_cfg,
        features=feature_cfg,
        extras=extras_cfg,
        runlogger=runlogger_cfg,
        source_paths={
            "environment": str(environment_path),
            "execution": str(execution_path),
            "assets": str(assets_path),
            "strategy": str(strategy_path),
            "risk": str(risk_wrapper_path),
            "behaviour": str(behaviour_path),
            "features": str(features_path),
            "extras": str(extras_path),
            "runlogger": str(runlogger_path),
        },
    )
    logger.info("Loaded GlobalConfig: %s", global_cfg.summary())
    return global_cfg


def load_all_configs_into_global() -> GlobalConfig:
    default_profile = CONFIG_ROOT / "profiles" / "sim.yaml"
    if default_profile.exists():
        return load_global_config_from_profile(str(default_profile))
    return load_global_config()


def validate_global_config(config: GlobalConfig) -> None:
    """
    Placeholder for future cross-config validation.
    """
    logger.debug("GlobalConfig validated: %s", config.summary())


def merge_dicts_for_ui(default: Dict, user_override: Dict) -> Dict:
    merged = dict(default)
    merged.update(user_override or {})
    return merged


def global_config_summary(global_config: GlobalConfig) -> Dict[str, object]:
    return {
        "environment": {
            "mode": global_config.environment.mode,
            "timezone": global_config.environment.timezone,
        },
        "risk_policy": global_config.risk.policy_type,
        "behaviour_enabled": global_config.behaviour.enabled,
        "strategies": global_config.strategy.enabled_strategies,
        "assets": list(global_config.assets.assets.keys()),
        "features_enabled": global_config.features.enabled,
        "model_features_enabled": global_config.features.model_features.enabled,
        "model_scaling_type": global_config.features.model_features.scaling.type,
        "extras_enabled": global_config.extras.enabled,
        "extras_datasets": [d.name for d in global_config.extras.get_enabled_datasets()],
        "runlogger_enabled": global_config.runlogger.enabled,
        "runlogger_base_dir": global_config.runlogger.base_dir,
    }


def _resolve_path(include_path: str, profile_path: Path) -> Path:
    candidate = Path(include_path)
    if candidate.is_absolute():
        return candidate
    # Resolve relative to profile file first, then project root.
    resolved = (profile_path.parent / candidate).resolve()
    if resolved.exists():
        return resolved
    return (PROJECT_ROOT / candidate).resolve()


def _build_risk_config(include_path: Path) -> RiskConfig:
    data = load_yaml(str(include_path))
    risk_data = data.get("risk", data)
    policy_type = risk_data.get("policy_type") or risk_data.get("type") or include_path.stem
    raw_policy_path = risk_data.get("policy_path")
    if raw_policy_path:
        candidate = Path(raw_policy_path)
        if candidate.is_absolute():
            resolved_policy_path = candidate
        else:
            relative_candidate = (include_path.parent / candidate).resolve()
            if relative_candidate.exists():
                resolved_policy_path = relative_candidate
            else:
                resolved_policy_path = (PROJECT_ROOT / candidate).resolve()
    else:
        resolved_policy_path = include_path
    return RiskConfig(policy_type=policy_type, policy_path=str(resolved_policy_path))


def load_global_config_from_profile(profile_path: str) -> GlobalConfig:
    profile = load_profile(profile_path)
    includes = profile.includes
    profile_path_obj = Path(profile_path).resolve()

    env_path = _resolve_path(includes.environment, profile_path_obj)
    exec_path = _resolve_path(includes.execution, profile_path_obj)
    assets_path = _resolve_path(includes.assets, profile_path_obj)
    strategy_path = _resolve_path(includes.strategy, profile_path_obj)
    risk_path = _resolve_path(includes.risk, profile_path_obj)
    behaviour_path = _resolve_path(includes.behaviour, profile_path_obj)
    features_path = _resolve_path(includes.features, profile_path_obj)
    extras_path = _resolve_path(includes.extras, profile_path_obj)
    runlogger_path = _resolve_path(includes.runlogger, profile_path_obj)

    environment_cfg = _load_environment_config(env_path)
    execution_cfg = _load_execution_config(exec_path)
    assets_cfg = _load_asset_config(assets_path)
    strategy_cfg = _load_strategy_config(strategy_path)
    risk_cfg = _build_risk_config(risk_path)
    behaviour_cfg = _load_behaviour_config(behaviour_path)
    feature_cfg = load_feature_config(str(features_path))
    extras_cfg = load_extras_config(str(extras_path))
    runlogger_cfg = load_runlogger_config(str(runlogger_path))

    global_cfg = GlobalConfig(
        environment=environment_cfg,
        execution=execution_cfg,
        assets=assets_cfg,
        risk=risk_cfg,
        behaviour=behaviour_cfg,
        strategy=strategy_cfg,
        features=feature_cfg,
        extras=extras_cfg,
        runlogger=runlogger_cfg,
        source_paths={
            "profile": str(profile_path_obj),
            "environment": str(env_path),
            "execution": str(exec_path),
            "assets": str(assets_path),
            "strategy": str(strategy_path),
            "risk": str(risk_path),
            "behaviour": str(behaviour_path),
            "features": str(features_path),
            "extras": str(extras_path),
            "runlogger": str(runlogger_path),
        },
    )

    logger.info(
        "GLOBAL_CONFIG_FROM_PROFILE | profile=%s | env=%s | risk_policy=%s | strategies=%s | behaviour_enabled=%s",
        profile.name,
        environment_cfg.mode,
        risk_cfg.policy_type,
        strategy_cfg.enabled_strategies,
        behaviour_cfg.enabled,
    )
    return global_cfg
