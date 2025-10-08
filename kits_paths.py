# -*- coding: utf-8 -*-
"""
Zentrale Pfad- und Config-Helfer für KITS.
Lege diese Datei ins Projekt-Root (dort wo Ordner `data_pipeline/` und `ml/` liegen).

Bietet:
- p(key): gibt wichtige Pfade zurück (features_dir, models_root, reports_dir, regime_cfg)
- cfg_load_regime(path=None): lädt regime.yaml (Default: ml/regime/configs/regime.yaml)
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import sys

try:
    import yaml  # PyYAML
except Exception:
    yaml = None

# Projekt-Root heuristisch = Ordner, in dem diese Datei liegt
PROJECT_ROOT = Path(__file__).resolve().parent

# Standardpfade gemäß deiner aktuellen Struktur
FEATURES_DIR = PROJECT_ROOT / "data_pipeline" / "data" / "features"
MODELS_ROOT  = PROJECT_ROOT / "ml" / "regime" / "models"
REPORTS_DIR  = PROJECT_ROOT / "ml" / "regime"  # hier landen CSV/Reports
REGIME_CFG   = PROJECT_ROOT / "ml" / "regime" / "configs" / "regime.yaml"

_PATHS = {
    "project_root": PROJECT_ROOT,
    "features_dir": FEATURES_DIR,
    "models_root":  MODELS_ROOT,
    "reports_dir":  REPORTS_DIR,
    "regime_cfg":   REGIME_CFG,
}

def p(key: str) -> Path:
    """Schneller Zugriff auf Standardpfade."""
    if key not in _PATHS:
        raise KeyError(f"Unbekannter Pfadschlüssel: {key} (verfügbar: {list(_PATHS.keys())})")
    return _PATHS[key]

def _ensure_dirs():
    for k in ("features_dir", "models_root", "reports_dir"):
        p(k).mkdir(parents=True, exist_ok=True)

def cfg_load_regime(config_path: str | os.PathLike | None = None) -> dict:
    """
    Lädt regime.yaml. Wenn kein Pfad übergeben, wird der Standard genommen.
    Fällt auf leere/minimale Defaults zurück, wenn yaml nicht verfügbar ist.
    """
    _ensure_dirs()
    cfg_file = Path(config_path) if config_path else REGIME_CFG

    # Fallback-Minimalconfig, falls Datei fehlt oder PyYAML nicht installiert ist
    minimal = {
        "version": "v1",
        "assets": ["EURUSD"],
        "timeframes": ["5m"],
        "mode_filter": "BACKTEST",
        "feature_cols": [
            "atr_pips_14","atr_pips_28","roc5_pct","adx14","chop14","rsi14",
            "ema21","ema50","ema100","ema21_slope","ema50_slope",
            "spread_pips_eff","commission_pips","cost_pips_roundtrip","fee_burden_ratio",
            "dow","hour"
        ],
        "clustering": {
            "pca_components": 8,
            "min_cluster_size": 400,
            "min_samples": None,
            "cluster_selection_method": "eom",
            "metric": "euclidean"
        },
        "mapping_rules": {
            "vol_threshold_pips": 6,
            "bull_threshold_roc": 0.05,
            "bear_threshold_roc": -0.05,
            "adx_threshold": 18,
            "chop_threshold": 55
        },
        "thresholds": {
            "score": 0.62, "bull": 0.6, "range": 0.5, "vol": 0.55
        },
        "walkforward": {
            "min_train_months": 2,
            "test_months": 1
        },
        "classifier": {
            "algo": "disabled",   # "xgboost" | "logreg" | "disabled"
            "heads_algo": "logreg",
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss"
        }
    }

    if not cfg_file.exists() or yaml is None:
        return minimal

    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except Exception:
        return minimal

    # Sanfte Defaults mergen (nur Top-Level)
    for k, v in minimal.items():
        if k not in loaded:
            loaded[k] = v

    return loaded
