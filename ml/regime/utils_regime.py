# -*- coding: utf-8 -*-
"""
utils_regime.py
Zentrale Utilities für das Regime-Training & -Inference.
Wird von train_regime.py & Regime_analysis.py importiert (keine Duplikate!).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Daten laden / vorbereiten
# -----------------------------------------------------------------------------
def load_feature_frames(features_dir: str,
                        assets: List[str],
                        timeframes: List[str],
                        mode_filter: str | None = "BACKTEST") -> pd.DataFrame:
    """
    Lädt Feature-Parquets je Asset/TF aus `features_dir`.
    Erwartet Dateinamen: {ASSET}_{TF}_*.parquet (z. B. EURUSD_15m_BACKTEST.parquet).
    """
    features_path = Path(features_dir)
    if not features_path.exists():
        raise FileNotFoundError(f"features_dir existiert nicht: {features_path}")

    dfs = []
    for a in assets:
        for tf in timeframes:
            patt = f"{a}_{tf}_*.parquet"
            for p in features_path.glob(patt):
                if mode_filter and mode_filter not in p.stem:
                    pass  # tolerant: trotzdem laden
                df = pd.read_parquet(p)
                if "timestamp" not in df.columns:
                    raise ValueError(f"'timestamp' Spalte fehlt in {p}")
                df["asset"] = a
                df["tf"] = tf
                dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"Keine Feature-Parquets gefunden unter {features_path}. "
            f"Gesucht: {', '.join([f'{a}_{tf}_*.parquet' for a in assets for tf in timeframes])}"
        )

    out = pd.concat(dfs, ignore_index=True).sort_values("timestamp")
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    # timestamp auf tz-naiv UTC normalisieren (für Perioden)
    if pd.api.types.is_datetime64tz_dtype(out["timestamp"]):
        out["timestamp"] = out["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    elif not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)

    return out


def pick_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Wählt existierende Feature-Spalten, castet auf float."""
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].astype(float).copy()
    return X, cols

# -----------------------------------------------------------------------------
# Adaptive (human-free) Weak-Label Gates für Heads
# -----------------------------------------------------------------------------
def _group_quantile(s: pd.Series, q: float) -> float:
    s = s.dropna()
    if len(s) == 0: return np.nan
    return float(np.nanpercentile(s.values, q))

def compute_adaptive_thresholds(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Berechnet per (asset, tf) adaptive Schwellen für ATR/ADX/CHOP/Slope
    basierend auf Quantilen (Perzentilen). Liefert Weak Labels:
      WL_BULL, WL_RANGE, WL_VOL
    """
    mr = cfg.get("mapping_rules", {}) or {}
    vol_q    = float(mr.get("vol_percentile", 60))
    adx_q_lo = float(mr.get("adx_percentile_low", 40))
    chp_q_hi = float(mr.get("chop_percentile_high", 60))
    slp_q_hi = float(mr.get("slope_percentile_bull", 55))

    out = df.copy()
    grp = out.groupby(["asset","tf"], observed=True)
    thr = grp.agg(
        atr_thr   = ("atr_pips_14", lambda x: _group_quantile(x, vol_q)),
        adx_thr   = ("adx14",       lambda x: _group_quantile(x, adx_q_lo)) if "adx14" in out.columns else ("atr_pips_14", lambda x: np.nan),
        chop_thr  = ("chop14",      lambda x: _group_quantile(x, chp_q_hi)) if "chop14" in out.columns else ("atr_pips_14", lambda x: np.nan),
        slope_thr = ("ema21_slope", lambda x: _group_quantile(x, slp_q_hi)),
    ).reset_index()

    out = out.merge(thr, on=["asset","tf"], how="left")

    # Weak Labels
    out["WL_VOL"] = (out["atr_pips_14"] >= out["atr_thr"]).astype(int)

    if "adx14" in out.columns or "chop14" in out.columns:
        adx_ok  = (out["adx14"]  <= out["adx_thr"])  if "adx14"  in out.columns else False
        chop_ok = (out["chop14"] >= out["chop_thr"]) if "chop14" in out.columns else False
        out["WL_RANGE"] = (adx_ok | chop_ok).astype(int)
    else:
        fbr = out["fee_burden_ratio"].fillna(0).clip(0,3)/3.0
        slope = (out["ema21_slope"].abs() + out["ema50_slope"].abs())
        slope_norm = (slope / (slope.rolling(200, min_periods=5).mean() + 1e-9)).clip(0,2)/2.0
        out["WL_RANGE"] = ((0.6 * fbr + 0.4 * (1 - slope_norm)) >= 0.8).astype(int)

    out["WL_BULL"] = (out["ema21_slope"] >= out["slope_thr"]).astype(int)
    return out

# -----------------------------------------------------------------------------
# Mapping: Cluster -> Zielregime (QC/Diagnose; weiterhin verfügbar)
# -----------------------------------------------------------------------------
def map_clusters_to_target(df: pd.DataFrame,
                           cluster_labels: np.ndarray,
                           cfg: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Robust mapping (für QC):
      - Volatilität: bevorzugt `vol_percentile`, Fallback `vol_threshold_pips`.
      - Range: bevorzugt ADX/CHOP; Fallback fee_burden_ratio + ema-Slopes.
      - Bull/Bear: ema21_slope > bull-threshold/perzentil (hier 0 genutzt).
    """
    df = df.copy()
    df["cluster"] = cluster_labels
    mr = cfg.get("mapping_rules", {}) or {}

    if "vol_percentile" in mr:
        vol_thresh = np.nanpercentile(df["atr_pips_14"].values, int(mr.get("vol_percentile", 60)))
    else:
        vol_thresh = float(mr.get("vol_threshold_pips", 12.0))
    is_volatile = (df["atr_pips_14"] >= vol_thresh).astype(int)

    bull_threshold = float(mr.get("bull_threshold", 0.0))
    bull_flag = (df["ema21_slope"] > bull_threshold).astype(int)

    if "adx14" in df.columns or "chop14" in df.columns:
        adx_gate  = float(mr.get("adx_gate", 18))
        chop_gate = float(mr.get("chop_gate", 55))
        adx_ok  = (df["adx14"] <= adx_gate)  if "adx14" in df.columns  else False
        chop_ok = (df["chop14"] >= chop_gate) if "chop14" in df.columns else False
        is_rangeish = (adx_ok | chop_ok).astype(int)
    else:
        fbr = df["fee_burden_ratio"].fillna(0).clip(0, 3) / 3.0
        slope = (df["ema21_slope"].abs() + df["ema50_slope"].abs())
        slope_norm = (slope / (slope.rolling(200, min_periods=5).mean() + 1e-9)).clip(0, 2) / 2.0
        is_rangeish = ((0.6*fbr + 0.4*(1 - slope_norm)) >= float(mr.get("range_gate", 0.8))).astype(int)

    mapping: Dict = {}
    clusters = [c for c in np.unique(cluster_labels) if c != -1]
    for c in clusters:
        sub = df.index[df["cluster"] == c]
        if len(sub) < 50:
            mapping[c] = "Range_Quiet"; continue
        r = int(np.median(is_rangeish.loc[sub]))
        v = int(np.median(is_volatile.loc[sub]))
        b = int(np.median(bull_flag.loc[sub]))
        if r == 1:
            regime = "Range_Volatile" if v else "Range_Quiet"
        else:
            regime = ("Bull_" if b==1 else "Bear_") + ("Volatile" if v else "Quiet")
        mapping[c] = regime

    mapping[-1] = "Range_Quiet"
    labels = np.array([mapping.get(c, "Range_Quiet") for c in cluster_labels])
    return labels, mapping

# -----------------------------------------------------------------------------
# Walk-forward Splits (TZ-sicher)
# -----------------------------------------------------------------------------
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

def month_walkforward_splits(df: pd.DataFrame,
                             min_train_months: int = 6,
                             test_months: int = 1) -> List[Tuple[np.ndarray, np.ndarray, str, str]]:
    """
    Erzeugt Walk-Forward-Splits auf Monatsbasis.
    Rückgabe: Liste von (train_idx, test_idx, train_span_str, test_span_str).
    """
    df = df.copy()
    if not (is_datetime64_any_dtype(df["timestamp"]) or is_datetime64tz_dtype(df["timestamp"])):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    ts = df["timestamp"]
    if is_datetime64tz_dtype(ts):
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"])

    df["ym"] = df["timestamp"].dt.to_period("M")
    months = sorted(df["ym"].unique())
    splits: List[Tuple[np.ndarray, np.ndarray, str, str]] = []
    if len(months) < (min_train_months + test_months):
        return splits

    for i in range(min_train_months, len(months) - test_months + 1):
        train_months = months[i - min_train_months : i]
        test_window  = months[i : i + test_months]
        tr_idx = df["ym"].isin(train_months).values
        te_idx = df["ym"].isin(test_window).values
        splits.append((
            df.index[tr_idx].values,
            df.index[te_idx].values,
            f"{train_months[0]}→{train_months[-1]}",
            f"{test_window[0]}→{test_window[-1]}",
        ))
    df.drop(columns=["ym"], inplace=True)
    return splits

# -----------------------------------------------------------------------------
# JSON-sicheres Speichern von Artefakten
# -----------------------------------------------------------------------------
def _jsonable(obj):
    import numpy as np
    import pandas as pd
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, (np.bool_,)):     return bool(obj)
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        return [_jsonable(x) for x in list(obj)]
    if isinstance(obj, pd.Timestamp):    return obj.isoformat()
    if isinstance(obj, pd.Period):       return str(obj)
    if isinstance(obj, dict):            return {str(_jsonable(k)): _jsonable(v) for k, v in obj.items()}
    return str(obj)

def save_artifacts(artifacts_dir: str,
                   scaler: StandardScaler,
                   feature_cols: List[str],
                   mapping: Dict,
                   clf,
                   meta: Dict):
    """Speichert Scaler, Classifier, Feature-Liste, Mapping und Meta JSON-sicher."""
    from joblib import dump
    adir = Path(artifacts_dir)
    adir.mkdir(parents=True, exist_ok=True)

    mapping_jsonable = {str(int(k)): (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                        for k, v in mapping.items()}

    dump(scaler, adir / "scaler.joblib")
    dump(clf,    adir / "clf6.joblib")
    (adir / "feature_cols.json").write_text(json.dumps(list(feature_cols)))
    (adir / "cluster_mapping.json").write_text(json.dumps(mapping_jsonable))
    (adir / "meta.json").write_text(json.dumps(_jsonable(meta), indent=2))

# -----------------------------------------------------------------------------
# Optional: Drift
# -----------------------------------------------------------------------------
def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index zwischen zwei Verteilungen."""
    expected = expected.dropna()
    actual   = actual.dropna()
    qs = np.linspace(0, 1, bins + 1)
    cuts = expected.quantile(qs).unique()
    expected_bins = pd.cut(expected, bins=cuts, include_lowest=True)
    actual_bins   = pd.cut(actual,   bins=cuts, include_lowest=True)
    e_counts = expected_bins.value_counts(normalize=True)
    a_counts = actual_bins.value_counts(normalize=True).reindex(e_counts.index).fillna(1e-6)
    e_counts = e_counts.replace(0, 1e-6)
    return float(np.sum((a_counts - e_counts) * np.log(a_counts / e_counts)))
