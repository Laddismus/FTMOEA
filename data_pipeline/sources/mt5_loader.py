# sources/mt5_loader.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def init_mt5(login: int, password: str, server: str, path: str = None):
    if not mt5.initialize(path, login=login, password=password, server=server):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    print("MT5 initialized")

def shutdown_mt5():
    mt5.shutdown()

def _pip_meta(symbol: str):
    """Return (point, points_per_pip, pip_value_per_lot_usd) using MT5 symbol info."""
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"MT5 symbol_info({symbol}) is None")
    point = info.point  # smallest price step
    # Heuristik: Forex hat meist 5 Nachkommastellen (0.00001), Pip = 10 points
    digits = info.digits
    points_per_pip = 10 if digits in (3, 5) else 1  # e.g. JPY pairs often digits=3 -> pip=0.01 (10 points)
    # pipValuePerLot in USD (falls verfügbar); sonst Näherung über tick_value
    pip_value = None
    try:
        # trade_tick_value ist oft der $-Wert eines ticks (point) pro lot
        tick_val = info.trade_tick_value if info.trade_tick_value else None
        if tick_val:
            pip_value = tick_val * points_per_pip
    except Exception:
        pass
    return point, points_per_pip, pip_value

def fetch_mt5(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    tf_map = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1
    }
    if timeframe not in tf_map:
        raise ValueError(f"Unsupported TF {timeframe}")

    point, points_per_pip, pip_value_per_lot = _pip_meta(symbol)

    start_dt = datetime.fromisoformat(start) if " " in start else datetime.fromisoformat(start + " 00:00:00")
    end_dt   = datetime.fromisoformat(end)   if " " in end   else datetime.fromisoformat(end + " 23:59:59")

    rates = mt5.copy_rates_range(symbol, tf_map[timeframe], start_dt, end_dt)
    if rates is None:
        raise RuntimeError(f"Could not fetch MT5 data for {symbol}")

    df = pd.DataFrame(rates)
    # MT5 columns: time, open, high, low, close, tick_volume, spread (points), real_volume
    df.rename(columns={
        "time":"timestamp", "open":"open", "high":"high",
        "low":"low", "close":"close", "tick_volume":"volume", "spread":"spread_points"
    }, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    # Spread in pips
    df["spread_pips"] = (df["spread_points"] / points_per_pip).astype(float)

    # Bid/Ask Approx:
    # Bei den meisten Brokern sind OHLC bid-basiert; ask ≈ close + spread_points*point
    df["bid"] = df["close"]
    df["ask"] = df["close"] + df["spread_points"] * point

    # Meta
    df["mt5_point"] = point
    df["mt5_points_per_pip"] = points_per_pip
    df["mt5_pip_value_per_lot_usd"] = pip_value_per_lot if pip_value_per_lot is not None else float("nan")

    return df[[
        "timestamp","open","high","low","close","volume","bid","ask",
        "spread_points","spread_pips","mt5_point","mt5_points_per_pip","mt5_pip_value_per_lot_usd"
    ]]
