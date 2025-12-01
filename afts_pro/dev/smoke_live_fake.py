from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from afts_pro.broker.fake import FakeBroker  # noqa: E402
from afts_pro.core.live_engine import LiveEngine, LiveOrder  # noqa: E402
from afts_pro.core.live_runner import LiveConfig, LiveRunner  # noqa: E402


@dataclass
class SmokeLiveConfig:
    symbol: str = "TEST"
    steps: int = 20
    poll_interval: float = 0.0
    output_dir: Path = Path("artifacts/dev_smoke_live")


class SmokeEngine(LiveEngine):
    """
    Minimal engine that always emits a small entry order for demo purposes.
    """

    def __init__(self, symbol: str):
        super().__init__(symbol=symbol)
        self._step = 0

    def process_live_tick(self, price, position):
        self._step += 1
        # emit a single market entry each tick
        return [LiveOrder(kind="entry", symbol=price.symbol, side="long", size=0.01)]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = SmokeLiveConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    broker = FakeBroker()
    engine = SmokeEngine(symbol=cfg.symbol)
    live_cfg = LiveConfig(symbol=cfg.symbol, poll_interval_sec=cfg.poll_interval, max_steps=cfg.steps, use_system_gate=False)
    runner = LiveRunner(live_cfg, broker, engine)
    logging.info("[SMOKE-LIVE] Starting live loop | steps=%s", cfg.steps)
    runner.run()
    # collect orders from broker state
    orders_log = []
    for pos in broker.state.positions.values():
        orders_log.append({"symbol": pos.symbol, "size": pos.size, "entry_price": pos.entry_price, "side": pos.side})
    out_path = cfg.output_dir / "orders.json"
    out_path.write_text(json.dumps(orders_log, indent=2))
    logging.info("[SMOKE-LIVE] Done | orders_logged=%d | out=%s", len(orders_log), out_path)


if __name__ == "__main__":
    main()
