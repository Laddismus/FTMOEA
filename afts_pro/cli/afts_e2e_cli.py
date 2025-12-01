from __future__ import annotations

import argparse
import logging
from pathlib import Path

from afts_pro.core.e2e_runner import E2ESimConfig, run_e2e_sim

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="AFTS-PRO E2E SIM RL Acceptance Runner")
    parser.add_argument("--config", default="configs/modes/sim_e2e_acceptance.yaml", help="E2E SIM config path.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = E2ESimConfig(config_path=args.config)
    result = run_e2e_sim(cfg)
    logger.info(
        "E2E SIM Summary | trades=%d | equity_start=%.2f | equity_end=%.2f | rl=%s",
        result.num_trades,
        result.equity_start,
        result.equity_end,
        result.has_rl_signals,
    )


if __name__ == "__main__":
    main()
