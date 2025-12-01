from __future__ import annotations

import sys
import logging
from datetime import datetime
from pathlib import Path

PROFILE_NAME = "orb_eurusd_exitagent_v1"
PROFILES_PATH = Path("configs/train_profiles.yaml")
SMOKE_OUTPUT_ROOT = Path("models/dev_smoke_train/exit")

# Ensure src is on path when running directly
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml  # noqa: E402
from afts_pro.core.train_controller import TrainController, TrainJobConfig  # noqa: E402


def load_profile(name: str) -> dict:
    data = yaml.safe_load(PROFILES_PATH.read_text())
    profiles = data.get("profiles", {})
    if name not in profiles:
        raise ValueError(f"Profile {name} not found in {PROFILES_PATH}")
    return profiles[name]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    profile = load_profile(PROFILE_NAME)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SMOKE_OUTPUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    job_cfg = TrainJobConfig(
        agent_type=profile["agent_type"],
        env_config_path=str(profile["env_config"]),
        agent_config_path=str(profile["agent_config"]),
        output_dir=str(out_dir),
        seed=43,
    )
    logging.info("[SMOKE-TRAIN] Starting exit smoke run | profile=%s | out=%s", PROFILE_NAME, out_dir)
    controller = TrainController()
    result = controller.run_train_job(job_cfg)
    logging.info(
        "[SMOKE-TRAIN] Done | episodes=%s | mean_return=%.4f | best_return=%.4f | checkpoint_dir=%s",
        result.episodes,
        result.mean_return,
        result.best_return,
        result.extra.get("checkpoint_dir"),
    )


if __name__ == "__main__":
    main()
