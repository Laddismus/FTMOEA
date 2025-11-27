from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from afts_pro.rl.rl_inference import RLInferenceHook, ObservationBuilder
from afts_pro.rl.risk_agent import RiskAgent, RiskAgentConfig
from afts_pro.rl.exit_agent import ExitAgent, ExitAgentConfig
from afts_pro.rl.types import RLObsSpec, ActionSpec

logger = logging.getLogger(__name__)


def _load_agent(path: str, agent_type: str, obs_spec: RLObsSpec):
    path_obj = Path(path)
    if agent_type == "risk":
        cfg_path = path_obj / "risk_agent_config.yaml" if path_obj.is_dir() else path_obj
        cfg = RiskAgentConfig(**yaml.safe_load(cfg_path.read_text()))
        agent = RiskAgent(cfg, obs_spec, ActionSpec(action_type="continuous", shape=(1,)))
        if path_obj.is_dir():
            agent.load(path_obj)
        return agent
    cfg_path = path_obj / "exit_agent_config.yaml" if path_obj.is_dir() else path_obj
    cfg = ExitAgentConfig(**yaml.safe_load(cfg_path.read_text()))
    agent = ExitAgent(cfg, obs_spec, ActionSpec(action_type="discrete", num_actions=cfg.n_actions))
    if path_obj.is_dir():
        agent.load(path_obj)
    return agent


def integrate_rl_inference(
    *,
    use_risk_agent: bool,
    use_exit_agent: bool,
    risk_agent_path: Optional[str],
    exit_agent_path: Optional[str],
    obs_spec: RLObsSpec,
) -> Optional[RLInferenceHook]:
    risk_agent = None
    exit_agent = None
    if use_risk_agent and risk_agent_path:
        try:
            risk_agent = _load_agent(risk_agent_path, "risk", obs_spec)
            logger.info("Loaded RiskAgent from %s", risk_agent_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load RiskAgent: %s", exc)
    if use_exit_agent and exit_agent_path:
        try:
            exit_agent = _load_agent(exit_agent_path, "exit", obs_spec)
            logger.info("Loaded ExitAgent from %s", exit_agent_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load ExitAgent: %s", exc)

    if not risk_agent and not exit_agent:
        return None
    obs_builder = ObservationBuilder(obs_spec=obs_spec)
    return RLInferenceHook(risk_agent=risk_agent, exit_agent=exit_agent, obs_builder=obs_builder)
