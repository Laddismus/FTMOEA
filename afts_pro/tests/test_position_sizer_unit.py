from afts_pro.exec.position_sizer import PositionSizer, PositionSizerConfig


def test_basic_sizing_from_agent_risk():
    cfg = PositionSizerConfig()
    sizer = PositionSizer(cfg)
    res = sizer.compute_position_size(
        symbol="ETH",
        side="long",
        entry_price=100.0,
        sl_price=99.0,
        equity=10000.0,
        agent_risk_pct=1.0,
    )
    assert res.size == 100.0
    assert res.effective_risk_pct == 1.0


def test_min_max_risk_clamping():
    cfg = PositionSizerConfig(min_risk_pct=0.0, max_risk_pct=2.0)
    sizer = PositionSizer(cfg)
    res = sizer.compute_position_size("ETH", "long", 100, 99, 10000, -1.0)
    assert res.effective_risk_pct == 0.0
    res2 = sizer.compute_position_size("ETH", "long", 100, 99, 10000, 10.0)
    assert res2.effective_risk_pct == 2.0


def test_max_risk_per_trade_cap():
    cfg = PositionSizerConfig(max_risk_per_trade_pct=1.0)
    sizer = PositionSizer(cfg)
    res = sizer.compute_position_size("ETH", "long", 100, 99, 10000, 5.0)
    assert "max_risk_per_trade" in res.capped_by
    assert res.effective_risk_pct <= 5.0


def test_default_sl_via_atr():
    cfg = PositionSizerConfig(default_sl_atr_factor=1.5)
    sizer = PositionSizer(cfg)
    res = sizer.compute_position_size("ETH", "long", 100, None, 10000, 1.0, atr=2.0)
    assert res.size > 0
    assert "default_sl_atr" in res.capped_by


def test_no_atr_and_no_sl_returns_zero():
    cfg = PositionSizerConfig()
    sizer = PositionSizer(cfg)
    res = sizer.compute_position_size("ETH", "long", 100, None, 10000, 1.0, atr=None)
    assert res.size == 0
    assert "missing_sl_info" in res.capped_by
