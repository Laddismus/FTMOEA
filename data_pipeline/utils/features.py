# utils/features.py
import pandas as pd
import numpy as np

def _adx(df, period=14):
    # Wilder's ADX in Preis-Einheiten
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm  = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0

    tr1 = (high - low)
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    pdi = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-9))
    mdi = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-9))
    dx  = 100 * ( (pdi - mdi).abs() / (pdi + mdi + 1e-9) )
    adx = dx.rolling(period).mean()
    return adx

def _choppiness(df, period=14):
    # Choppiness Index (0..100), hoch = seitwärts/choppy
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = (high - low)
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(1).sum()  # TR Summe über Fenster
    tr_sum = tr.rolling(period).sum()
    highest = high.rolling(period).max()
    lowest  = low.rolling(period).min()
    denom = (highest - lowest).replace(0, np.nan)
    chop = 100 * np.log10(tr_sum / denom) / np.log10(period)
    return chop


def _atr_pips(df: pd.DataFrame, window: int = 14) -> pd.Series:
    # TR in Preis-Einheiten
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    # Preis-Einheiten -> Pips: benutze points_per_pip*point
    # point kann je Zeile leicht variieren; wir nehmen die Spalte mt5_point falls vorhanden, sonst schätzen über bid/ask
    if "mt5_point" in df.columns and "mt5_points_per_pip" in df.columns:
        pip_value = df["mt5_point"] * df["mt5_points_per_pip"]
        atr_pips = atr / (pip_value.replace(0, np.nan))
    else:
        # Fallback: schätze Pipgröße über Preisauflösung (z. B. 1e-4)
        # Das ist nur Notlösung für Quellen ohne MT5-Meta, z. B. Crypto
        price = df["close"]
        # Heuristik: wenn Preis < 10 -> pip ~ 0.0001; wenn Preis > 1000 -> pip ~ 0.1; sonst 0.01/0.001
        approx_pip = np.where(price > 1000, 0.1, np.where(price > 10, 0.01, 0.0001))
        atr_pips = atr / approx_pip
    return atr_pips

def add_features(
    df: pd.DataFrame,
    instrument_cfg: dict
) -> pd.DataFrame:
    """
    instrument_cfg:
      type: forex | index | crypto
      commission:
        kind: usd_per_lot_roundtrip | perc
        value: 3.0          # if kind=usd_per_lot_roundtrip
        pip_value_per_lot_usd: 10.0   # optional override; else use mt5 column
        perc: 0.0002        # if kind=perc (0.02%)
    """
    df = df.copy()

    # === ATR in Pips ===
    df["atr_pips_14"] = _atr_pips(df, window=14)
    df["atr_pips_28"] = _atr_pips(df, window=28)

    # === Momentum / RSI (preisnormalisiert bleibt) ===
    df["roc5_pct"] = df["close"].pct_change(5) * 100

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # === Trend EMAs ===
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema100"] = df["close"].ewm(span=100).mean()
    df["ema21_slope"] = df["ema21"].diff()
    df["ema50_slope"] = df["ema50"].diff()

    # === Spread (Pips) ===
    if "spread_pips" in df.columns:
        df["spread_pips_eff"] = df["spread_pips"].astype(float)
    elif "bid" in df.columns and "ask" in df.columns and "mt5_point" in df.columns and "mt5_points_per_pip" in df.columns:
        # Backup-Berechnung
        df["spread_pips_eff"] = ( (df["ask"] - df["bid"]) / (df["mt5_point"] * df["mt5_points_per_pip"]) )
    else:
        # Fallback: Prozent-Spread (z. B. bei Krypto) grob in "pseudo-pips" mappen
        df["spread_pips_eff"] = np.nan

    # === Kommission in Pips (pro Roundtrip) ===
    comm = instrument_cfg.get("commission", {})
    kind = comm.get("kind", "usd_per_lot_roundtrip" if instrument_cfg.get("type")=="forex" else "perc")

    if kind == "usd_per_lot_roundtrip":
        usd = float(comm.get("value", 3.0))  # $/lot roundtrip
        pip_val_override = comm.get("pip_value_per_lot_usd", None)
        if pip_val_override is not None:
            pip_value_per_lot = float(pip_val_override)
        else:
            # Versuch, aus MT5 Spalte zu lesen; sonst Default 10 $
            pip_value_per_lot = df["mt5_pip_value_per_lot_usd"].fillna(10.0)

        df["commission_pips"] = usd / pip_value_per_lot
    elif kind == "perc":
        # Für Krypto: pro Roundtrip (Maker/Taker) in Prozent z. B. 0.04% → approx in pips relativ zu Preis nicht sinnvoll
        # Wir bilden Kommission in Preis-Einheiten ab und teilen durch pip-size → pipschätzung
        perc = float(comm.get("perc", 0.0004))  # 0.04%
        # Preis-Kommission je Roundtrip ~ perc * close
        price_comm = df["close"] * perc
        # pipsize
        if "mt5_point" in df.columns and "mt5_points_per_pip" in df.columns:
            pipsize = df["mt5_point"] * df["mt5_points_per_pip"]
        else:
            # Heuristik wie oben
            price = df["close"]
            approx_pip = np.where(price > 1000, 0.1, np.where(price > 10, 0.01, 0.0001))
            pipsize = approx_pip
        df["commission_pips"] = price_comm / (pipsize.replace(0, np.nan))
    else:
        df["commission_pips"] = 0.0

    # === fee_burden_ratio ===
    df["cost_pips_roundtrip"] = df["spread_pips_eff"].fillna(0) + df["commission_pips"].fillna(0)
    df["fee_burden_ratio"] = df["cost_pips_roundtrip"] / (df["atr_pips_14"].replace(0, np.nan))

    # === Context ===
    df["dow"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour

    # ADX & Choppiness:
    df["adx14"]  = _adx(df, 14)
    df["chop14"] = _choppiness(df, 14)

    return df
