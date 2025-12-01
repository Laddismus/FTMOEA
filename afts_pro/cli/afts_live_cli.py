from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from afts_pro.broker.fake import FakeBroker
from afts_pro.core.live_engine import LiveEngine
from afts_pro.core.live_runner import LiveConfig, LiveRunner

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="AFTS-PRO LIVE mode skeleton")
    parser.add_argument("--config", default="configs/modes/live.yaml", help="Path to live config YAML.")
    parser.add_argument("--broker", default="fake", help="Broker type (only 'fake' supported in skeleton).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg_data = yaml.safe_load(Path(args.config).read_text()) if Path(args.config).exists() else {}
    live_cfg = LiveConfig(**cfg_data)

    broker = FakeBroker()
    engine = LiveEngine(symbol=live_cfg.symbol)
    runner = LiveRunner(live_cfg, broker, engine)
    runner.run()


if __name__ == "__main__":
    main()
