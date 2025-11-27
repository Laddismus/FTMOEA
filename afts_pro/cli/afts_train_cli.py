from __future__ import annotations

import argparse
import logging
from pathlib import Path
import yaml

from afts_pro.core.train_controller import TrainController, TrainJobConfig

logger = logging.getLogger(__name__)


def _load_train_profile(profile_name: str, profiles_path: str) -> dict:
    data = yaml.safe_load(Path(profiles_path).read_text())
    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown train profile: {profile_name}")
    return profiles[profile_name]


def main() -> None:
    parser = argparse.ArgumentParser(description="AFTS-PRO TRAIN mode CLI")
    parser.add_argument("--train-profile", default="risk_default", help="Train profile name.")
    parser.add_argument("--profiles-path", default="configs/train_profiles.yaml", help="Path to train profiles yaml.")
    parser.add_argument("--env-config", help="Override env config path.")
    parser.add_argument("--agent-config", help="Override agent config path.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--agent-type", help="Override agent type (risk/exit).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    profile = _load_train_profile(args.train_profile, args.profiles_path)
    agent_type = args.agent_type or profile.get("agent_type", "risk")
    env_config = args.env_config or profile.get("env_config")
    agent_config = args.agent_config or profile.get("agent_config")
    output_root = args.output_dir or profile.get("output_root", "models/train")
    output_dir = Path(output_root) / args.train_profile

    job_cfg = TrainJobConfig(
        agent_type=agent_type,
        env_config_path=str(env_config),
        agent_config_path=str(agent_config),
        output_dir=str(output_dir),
        seed=None,
    )
    controller = TrainController()
    result = controller.run_train_job(job_cfg)
    logger.info("TRAIN completed | agent_type=%s | mean_return=%.4f | best_return=%.4f", result.agent_type, result.mean_return, result.best_return)


if __name__ == "__main__":
    main()
