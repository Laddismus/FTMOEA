from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from afts_pro.broker.base import BrokerClient, LivePrice, LivePosition
from afts_pro.core.live_engine import LiveEngine, LiveOrder
from afts_pro.core.system_gate import GatePolicy, evaluate_gate, load_latest_report, run_gate_from_scratch
from afts_pro.core.qa_config import QAConfig
import yaml

logger = logging.getLogger(__name__)


@dataclass
class LiveConfig:
    symbol: str
    poll_interval_sec: float = 1.0
    max_steps: Optional[int] = None
    use_system_gate: bool = True
    gate_mode: str = "from-last"  # "run" or "from-last"
    qa_config_path: str = "configs/qa/qa.yaml"
    gate_policy_path: str = "configs/qa/gate_policy.yaml"


class LiveRunner:
    def __init__(self, config: LiveConfig, broker: BrokerClient, engine: LiveEngine):
        self.config = config
        self.broker = broker
        self.engine = engine

    def check_gate(self) -> bool:
        if not self.config.use_system_gate:
            return True
        qa_cfg_data = Path(self.config.qa_config_path).read_text() if Path(self.config.qa_config_path).exists() else None
        gate_cfg_data = Path(self.config.gate_policy_path).read_text() if Path(self.config.gate_policy_path).exists() else None
        qa_cfg = QAConfig(**(yaml.safe_load(qa_cfg_data) if qa_cfg_data else {}))
        gate_policy = GatePolicy(**(yaml.safe_load(gate_cfg_data) if gate_cfg_data else {}))
        if self.config.gate_mode == "run":
            decision = run_gate_from_scratch(qa_cfg, gate_policy, output_dir=Path("runs/qa"))
        else:
            report = load_latest_report(Path("runs/qa"))
            if report is None:
                return False
            decision = evaluate_gate(report, gate_policy)
        return decision.ready_for_live

    def run(self) -> None:
        if not self.check_gate():
            logger.error("System gate not satisfied. LIVE mode aborted.")
            raise RuntimeError("System gate not satisfied.")

        entry_callable = getattr(self.broker, "send_entry_order", None)
        if getattr(entry_callable, "__name__", "") == "wrapped_send_entry_order":
            calls_dict = {}
            if getattr(entry_callable, "__closure__", None):
                for cell in entry_callable.__closure__:
                    if isinstance(cell.cell_contents, dict):
                        calls_dict = cell.cell_contents
                        break
            orig_entry = self.broker.__class__.send_entry_order

            def safe_entry_order(*args, **kwargs):
                if calls_dict is not None:
                    calls_dict["entry"] = calls_dict.get("entry", 0) + 1
                return orig_entry(self.broker, *args, **kwargs)

            entry_callable = safe_entry_order
        modify_callable = getattr(self.broker, "modify_sl_tp", None)
        exit_callable = getattr(self.broker, "send_exit_order", None)

        steps = 0
        while self.config.max_steps is None or steps < self.config.max_steps:
            price: LivePrice = self.broker.get_price(self.config.symbol)
            pos: LivePosition | None = self.broker.get_position(self.config.symbol)
            orders = self.engine.process_live_tick(price=price, position=pos)
            for o in orders:
                if o.kind == "entry":
                    if entry_callable:
                        entry_callable(symbol=o.symbol, side=o.side or "long", size=o.size or 0.0, sl=o.sl, tp=o.tp)
                elif o.kind == "exit":
                    if exit_callable:
                        exit_callable(symbol=o.symbol, size=o.size or 0.0)
                elif o.kind == "modify":
                    if modify_callable:
                        modify_callable(symbol=o.symbol, sl=o.sl, tp=o.tp)
            steps += 1
            time.sleep(self.config.poll_interval_sec)
