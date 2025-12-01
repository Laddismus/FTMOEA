from pathlib import Path

import pytest

from afts_pro.broker.fake import FakeBroker
from afts_pro.core.live_engine import LiveEngine, LiveOrder
from afts_pro.core.live_runner import LiveConfig, LiveRunner


class DummyEngine(LiveEngine):
    def process_live_tick(self, price, position):
        return [
            LiveOrder(kind="entry", symbol=price.symbol, side="long", size=0.1, sl=None, tp=None),
        ]


def test_live_runner_runs_n_steps_with_fake_broker(monkeypatch):
    cfg = LiveConfig(symbol="TEST", poll_interval_sec=0.0, max_steps=3, use_system_gate=False)
    broker = FakeBroker()
    engine = DummyEngine(symbol="TEST")
    calls = {"entry": 0}

    def wrapped_send_entry_order(*args, **kwargs):
        calls["entry"] += 1
        return broker.send_entry_order(*args, **kwargs)

    monkeypatch.setattr(broker, "send_entry_order", wrapped_send_entry_order)

    runner = LiveRunner(cfg, broker, engine)
    runner.run()
    assert calls["entry"] == 3


def test_live_runner_respects_gate(monkeypatch):
    cfg = LiveConfig(symbol="TEST", poll_interval_sec=0.0, max_steps=1, use_system_gate=True)
    broker = FakeBroker()
    engine = DummyEngine(symbol="TEST")
    runner = LiveRunner(cfg, broker, engine)
    monkeypatch.setattr(runner, "check_gate", lambda: False)
    with pytest.raises(RuntimeError):
        runner.run()
