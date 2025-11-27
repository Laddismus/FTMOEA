from __future__ import annotations

import argparse
import logging
from pathlib import Path
import yaml

from afts_pro.rl.env import RLTradingEnv, load_env_config
from afts_pro.rl.risk_agent import RiskAgent, RiskAgentConfig
from afts_pro.rl.risk_training import TrainLoopConfig, train_risk_agent
from afts_pro.rl.types import ActionSpec, RLObsSpec

logger = logging.getLogger(__name__)


def _build_env(env_config_path: str) -> RLTradingEnv:
    cfg = load_env_config(env_config_path)
    event_stream = [{"equity": 1.0, "drawdown": 0.0, "dd_remaining": 1.0, "features": [0.0, 0.0, 0.0, 0.0]} for _ in range(200)]
    return RLTradingEnv(cfg, event_stream=event_stream)


def main() -> None:
    parser = argparse.ArgumentParser(description="RiskAgent training CLI")
    parser.add_argument("--env-config", default="configs/rl/env.yaml", help="Path to env config.")
    parser.add_argument("--agent-config", default="configs/rl/risk_agent.yaml", help="Path to agent config.")
    parser.add_argument("--output-dir", default="models/risk_agent/exp001", help="Directory to store checkpoints.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    env = _build_env(args.env_config)
    obs_spec = RLObsSpec(shape=env.obs_spec.shape, dtype="float32", as_dict=False)
    action_spec = ActionSpec(action_type="continuous", num_actions=None, shape=(1,))

    agent_cfg = RiskAgentConfig(**yaml.safe_load(Path(args.agent_config).read_text()))
    agent = RiskAgent(agent_cfg, obs_spec=obs_spec, action_spec=action_spec)
    train_cfg = TrainLoopConfig()
    summary = train_risk_agent(env, agent, train_cfg, checkpoint_dir=Path(args.output_dir))
    logger.info("Training complete | mean_return=%.4f | best_return=%.4f", summary.mean_return, summary.best_return)


if __name__ == "__main__":
    main()
