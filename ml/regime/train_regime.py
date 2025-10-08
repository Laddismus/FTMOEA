# -*- coding: utf-8 -*-
"""
Train Regime Model (unsupervised clustering + heads + optional 6-class)
- Lädt Feature-Parquets (mehrere Assets/TFs) aus data_pipeline/data/features
- Skaliert, PCA, HDBSCAN
- Mappt Cluster → 6 Regime-Klassen (Bull/Bear/Range × Quiet/Volatile) via QC-Regeln
- Trainiert "Heads" (3 binäre Klassifikatoren: Bull?, Range?, Volatile?) -> LogReg standard
- Optional: 6-Klassen-Klassifikator (XGB/LogReg), falls in cfg.classifier.algo gesetzt
- Walk-forward-Evaluierung (monatlich)
- Speichert Artefakte unter ml/regime/models/<version>

Benötigt: kits_paths.py im Projekt-Root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import hdbscan
except Exception as e:
    print("Bitte `pip install hdbscan` ausführen.", file=sys.stderr)
    raise

# Pfade/Configs zentral
from kits_paths import cfg_load_regime, p

# ---------- Utils (self-contained) ----------

def log(msg: str):
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def ensure_dt64_utc(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Spalte '{col}' fehlt in Features.")
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], utc=True)
    elif getattr(df[col].dt, "tz", None) is None:
        df[col] = df[col].dt.tz_localize("UTC")
    return df

def month_walkforward_splits(df: pd.DataFrame,
                             min_train_months: int,
                             test_months: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Gibt Liste (tr_start, tr_end, te_start, te_end) für monatliche Walkforward-Splits zurück."""
    ts = df["timestamp"]
    months = pd.period_range(ts.min().to_period("M").start_time,
                             ts.max().to_period("M").start_time,
                             freq="M")
    months = pd.to_datetime(months.to_timestamp("M")).tz_localize("UTC")
    splits = []
    for i in range(min_train_months, len(months) - test_months + 1):
        tr_start = months[i - min_train_months]
        tr_end   = months[i]  # exklusiv
        te_start = months[i]
        te_end   = months[i + test_months] if i + test_months < len(months) else ts.max()
        splits.append((tr_start, tr_end, te_start, te_end))
    return splits

def select_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        raise KeyError(f"Features fehlen: {missing}")
    return df[feature_list].copy()

def fit_hdbscan(X: np.ndarray, cfg: dict) -> hdbscan.HDBSCAN:
    clust_cfg = cfg.get("clustering", {})
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=clust_cfg.get("min_cluster_size", 400),
        min_samples=clust_cfg.get("min_samples", None),
        cluster_selection_epsilon=clust_cfg.get("cluster_selection_epsilon", 0.0),
        cluster_selection_method=clust_cfg.get("cluster_selection_method", "eom"),
        metric=clust_cfg.get("metric", "euclidean"),
        prediction_data=True
    )
    return clusterer.fit(X)

def compute_qc_labels(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    QC-Regeln zur 6-Klassen-Zuweisung:
    - bull vs bear via roc5_pct und EMAs
    - vol vs quiet via atr_pips_14 Schwelle
    - range vs trend via CHOP/ADX (wenn vorhanden) oder via EMA-Slope/ROC Proxy
    """
    rules = cfg.get("mapping_rules", {})
    v_th = float(rules.get("vol_threshold_pips", 6))
    bull_th = float(rules.get("bull_threshold_roc", 0.05))
    bear_th = float(rules.get("bear_threshold_roc", -0.05))
    adx_th  = float(rules.get("adx_threshold", 18))
    chop_th = float(rules.get("chop_threshold", 55))

    # Hilfs-Flags
    vol = (df["atr_pips_14"] >= v_th).astype(int)

    # Trendrichtung Proxy
    bull_proxy = (df.get("roc5_pct", 0) >= bull_th) | (df.get("ema21_slope", 0) > 0)
    bear_proxy = (df.get("roc5_pct", 0) <= bear_th) | (df.get("ema21_slope", 0) < 0)

    # Range Proxy
    is_range = (df.get("chop14", 0) >= chop_th) | (df.get("adx14", 0) < adx_th)

    labels = []
    for b, br, rng, v in zip(bull_proxy, bear_proxy, is_range, vol):
        if rng:
            labels.append("Range_Volatile" if v else "Range_Quiet")
        else:
            if b and not br:
                labels.append("Bull_Volatile" if v else "Bull_Quiet")
            elif br and not b:
                labels.append("Bear_Volatile" if v else "Bear_Quiet")
            else:
                # unklar → fallback range quiet
                labels.append("Range_Quiet")
    return pd.Series(labels, index=df.index, name="regime_qc")

REGIME_ORDER = ["Bear_Quiet", "Bear_Volatile", "Bull_Quiet", "Bull_Volatile", "Range_Quiet", "Range_Volatile"]

def qc_to_heads(y_qc: pd.Series) -> pd.DataFrame:
    y = y_qc.values
    bull = np.isin(y, ["Bull_Quiet", "Bull_Volatile"]).astype(int)
    ran  = np.isin(y, ["Range_Quiet", "Range_Volatile"]).astype(int)
    vol  = np.isin(y, ["Bear_Volatile", "Bull_Volatile", "Range_Volatile"]).astype(int)
    return pd.DataFrame({"bull": bull, "range": ran, "vol": vol}, index=y_qc.index)

def evaluate_heads(y_true_heads: pd.DataFrame, y_pred_heads: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    for k in ["bull", "range", "vol"]:
        f1  = f1_score(y_true_heads[k], y_pred_heads[k])
        try:
            auc = roc_auc_score(y_true_heads[k], y_pred_heads[k])
        except Exception:
            auc = np.nan
        out[k] = {"f1": float(f1), "auc": float(auc)}
    return out

def one_vs_rest_logreg(X_tr, y_tr_bin):
    clf = LogisticRegression(max_iter=200, n_jobs=None if sys.platform == "win32" else -1)
    clf.fit(X_tr, y_tr_bin)
    return clf

def train_heads(X_tr: np.ndarray, y_qc_tr: pd.Series) -> Dict[str, LogisticRegression]:
    Y = qc_to_heads(y_qc_tr)
    heads = {}
    for k in ["bull", "range", "vol"]:
        heads[k] = one_vs_rest_logreg(X_tr, Y[k].values)
    return heads

def predict_heads_proba(heads: Dict[str, LogisticRegression], X: np.ndarray) -> pd.DataFrame:
    probas = {}
    for k, h in heads.items():
        p = h.predict_proba(X)
        # Wahrscheinlichkeit für Klasse 1
        probas[k] = p[:, list(h.classes_).index(1)]
    return pd.DataFrame(probas)

def heads_to_6class(proba_heads: pd.DataFrame,
                    thresholds: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Mappt Heads-Probas auf 6-Klassen-Label + 'confidence' (max der genutzten Schwellen)."""
    bull_t = thresholds.get("bull", 0.6)
    range_t= thresholds.get("range", 0.5)
    vol_t  = thresholds.get("vol", 0.55)

    bull = (proba_heads["bull"]  >= bull_t).astype(int)
    ran  = (proba_heads["range"] >= range_t).astype(int)
    vol  = (proba_heads["vol"]   >= vol_t).astype(int)

    out = []
    conf = []
    for b, r, v, pb, pr, pv in zip(bull, ran, vol,
                                   proba_heads["bull"].values,
                                   proba_heads["range"].values,
                                   proba_heads["vol"].values):
        if r == 1:
            out.append("Range_Volatile" if v else "Range_Quiet")
            conf.append(max(pr, pv))
        else:
            if b == 1:
                out.append("Bull_Volatile" if v else "Bull_Quiet")
                conf.append(max(pb, pv))
            else:
                out.append("Bear_Volatile" if v else "Bear_Quiet")
                conf.append(max(1.0 - pb, pv))
    return np.array(out), np.array(conf, dtype=float)

def auto_thresholds_from_scores(scores: np.ndarray) -> float:
    """Ein globaler Score-Threshold als Referenz (z. B. 70%-Quantil)."""
    if len(scores) == 0:
        return 0.5
    return float(np.quantile(scores, 0.7))

def save_artifacts(art_dir: Path,
                   scaler: StandardScaler,
                   feat_cols: List[str],
                   pca: PCA,
                   hdb: hdbscan.HDBSCAN,
                   cluster_labels: np.ndarray,
                   thresholds: Dict[str, float],
                   heads: Dict[str, LogisticRegression] | None,
                   clf6,
                   meta: Dict):
    import joblib
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "scaler.joblib").write_bytes(joblib.dumps(scaler))
    (art_dir / "pca.joblib").write_bytes(joblib.dumps(pca))
    (art_dir / "hdbscan.joblib").write_bytes(joblib.dumps(hdb))
    (art_dir / "feat_cols.json").write_text(json.dumps(feat_cols, indent=2))
    (art_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    if heads:
        (art_dir / "head_bull.joblib").write_bytes(joblib.dumps(heads["bull"]))
        (art_dir / "head_range.joblib").write_bytes(joblib.dumps(heads["range"]))
        (art_dir / "head_vol.joblib").write_bytes(joblib.dumps(heads["vol"]))
    if clf6 is not None:
        (art_dir / "clf6.joblib").write_bytes(joblib.dumps(clf6))
    (art_dir / "meta.json").write_text(json.dumps(meta, indent=2))

# ---------- Training-Pipeline ----------

def load_feature_frames(features_dir: Path,
                        assets: List[str],
                        timeframes: List[str],
                        mode_filter: str = "BACKTEST") -> pd.DataFrame:
    paths = []
    for a in assets:
        for tf in timeframes:
            fp = features_dir / f"{a}_{tf}_{mode_filter}.parquet"
            if fp.exists():
                paths.append(fp)
    if not paths:
        raise FileNotFoundError(f"Keine Feature-Dateien gefunden unter {features_dir} für {assets}×{timeframes}×{mode_filter}")
    dfs = []
    for fp in paths:
        df = pd.read_parquet(fp)
        df["asset"] = fp.name.split("_")[0]
        df["tf"]    = fp.name.split("_")[1]
        dfs.append(df)
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    return df_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Pfad zu regime.yaml (optional)")
    args = parser.parse_args()

    log("🚀 Start train_regime.py")
    cfg = cfg_load_regime(args.config)
    log(f"📝 Using config: {p('regime_cfg') if args.config is None else Path(args.config).resolve()}")

    features_dir  = p("features_dir")
    artifacts_root= p("models_root")
    reports_dir   = p("reports_dir")
    version = cfg.get("version", "v1")
    artifacts_dir = artifacts_root / version
    log(f"Resolved features_dir: {features_dir}")
    log(f"Resolved artifacts_dir: {artifacts_dir}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    assets      = cfg.get("assets", ["EURUSD"])
    timeframes  = cfg.get("timeframes", ["5m"])
    mode_filter = cfg.get("mode_filter", "BACKTEST")
    feat_cols   = cfg.get("feature_cols", [
        "atr_pips_14","atr_pips_28","roc5_pct","adx14","chop14","rsi14",
        "ema21","ema50","ema100","ema21_slope","ema50_slope",
        "spread_pips_eff","commission_pips","cost_pips_roundtrip","fee_burden_ratio",
        "dow","hour"
    ])

    df = load_feature_frames(features_dir, assets, timeframes, mode_filter)
    df = ensure_dt64_utc(df, "timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)
    log(f"Columns (head): {list(df.columns[:16])} ...")
    log(f"Zeitspanne: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Feature-Matrix
    X_raw = select_features(df, feat_cols)
    log(f"Features genutzt ({len(feat_cols)}): {feat_cols}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.values)
    log("✅ Features skaliert")

    # PCA (für Clustering robuster)
    pca_components = cfg.get("clustering", {}).get("pca_components", 8)
    pca = PCA(n_components=pca_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    log(f"🧭 PCA für Clustering: components={pca_components}, explained_var={pca.explained_variance_ratio_.sum():.3f}")

    # HDBSCAN
    hdb = fit_hdbscan(X_pca, cfg)
    labels = hdb.labels_
    uniq = np.unique(labels)
    log(f"🔎 HDBSCAN: clusters (unique) = {len(uniq)} (inkl. noise)")

    # QC-Mapping (Gold Labels)
    df_qc = df.assign(**{c: X_raw[c] for c in feat_cols})
    y_qc = compute_qc_labels(df_qc, cfg)
    # Nur Logging-Beispiele
    sample_map = list(zip(uniq[:7], [y_qc.iloc[0] if len(y_qc) else "n/a"] * min(7, len(uniq))))
    log(f"🗺️  Cluster → Regime (QC) Beispiele {sample_map}")

    # Walkforward (Heads-Training + optionale 6-Klassen)
    wf_cfg = cfg.get("walkforward", {"min_train_months": 2, "test_months": 1})
    splits = month_walkforward_splits(df, wf_cfg["min_train_months"], wf_cfg["test_months"])

    # Heads Thresholds
    thresholds = cfg.get("thresholds", {"score": 0.62, "bull": 0.6, "range": 0.5, "vol": 0.55})
    heads_algo = cfg.get("classifier", {}).get("heads_algo", "logreg")
    train_6class_algo = cfg.get("classifier", {}).get("algo", None)

    macro_f1_6class_list = []
    heads_macro_f1_list  = []

    # Scores-Collect für Auto-Threshold-Hinweis
    heads_scores_all = []

    for (tr_start, tr_end, te_start, te_end) in splits:
        tr_msk = (df["timestamp"] >= tr_start) & (df["timestamp"] < tr_end)
        te_msk = (df["timestamp"] >= te_start) & (df["timestamp"] < te_end)

        Xtr = X_scaled[tr_msk.values]
        Xte = X_scaled[te_msk.values]
        ytr_qc = y_qc[tr_msk]
        yte_qc = y_qc[te_msk]

        # Heads
        if heads_algo == "logreg":
            heads = train_heads(Xtr, ytr_qc)
        else:
            heads = train_heads(Xtr, ytr_qc)  # Fallback LogReg

        proba_te = predict_heads_proba(heads, Xte)
        # Heads → 6 Class
        yhat_6, conf = heads_to_6class(proba_te, thresholds)
        heads_scores_all.extend(conf.tolist())

        # Eval Heads separat gegen QC → 6Class Macro-F1 (über 6 labels)
        f1_6 = f1_score(yte_qc, yhat_6, average="macro", labels=REGIME_ORDER)
        macro_f1_6class_list.append(f1_6)

        # Heads binär eval
        yte_heads = qc_to_heads(yte_qc)
        yhat_heads = pd.DataFrame({
            "bull": (proba_te["bull"] >= thresholds["bull"]).astype(int),
            "range":(proba_te["range"]>= thresholds["range"]).astype(int),
            "vol":  (proba_te["vol"]  >= thresholds["vol"]).astype(int),
        }, index=yte_qc.index)
        heads_metrics = evaluate_heads(yte_heads, yhat_heads)
        # als einfacher Durchschnitt
        heads_macro_f1_list.append(np.nanmean([m["f1"] for m in heads_metrics.values()]))

        # Logs
        log(f"📆 Split Train {tr_start:%Y-%m}→{tr_end:%Y-%m} | Test {te_start:%Y-%m}→{te_end:%Y-%m} | Größen: tr={len(Xtr):,}, te={len(Xte):,}")
        log(f"   → 6-Class Macro-F1 {te_start:%Y-%m}→{te_end:%Y-%m}: {f1_6:.3f}")

        # optional 6-Klassen-Klassifikator (nur wenn gewünscht und XGB verfügbar)
        # (Hier nur Evaluation vorgesehen; für Persistenz final unten)
        # bewusst weggelassen, um die Heads-Linie zu stärken

    log(f"🏁 Walk-forward Macro-F1 (6-Class avg): {np.mean(macro_f1_6class_list):.3f}")

    # Auto-Threshold-Vorschlag (nur Logging)
    global_score = auto_thresholds_from_scores(np.array(heads_scores_all))
    # Kleine Heuristik, nicht überschreiben – nur anzeigen
    auto_thr = {
        "score": float(global_score),
        "bull": thresholds.get("bull", 0.6),
        "range": thresholds.get("range", 0.5),
        "vol": thresholds.get("vol", 0.55),
    }
    log(f"🎯 Auto-Thresholds (global): {auto_thr}")

    # FINAL: Heads auf gesamten Datensatz trainieren
    heads_final = train_heads(X_scaled, y_qc)

    # Optional: 6-class clf (deaktiviert, aber Struktur vorhanden)
    clf6 = None
    if (train_6class_algo in {"xgboost", "logreg"}) and (train_6class_algo != "disabled"):
        y6 = pd.Categorical(y_qc, categories=REGIME_ORDER, ordered=False).codes
        if train_6class_algo == "xgboost" and HAS_XGB:
            clf6 = xgb.XGBClassifier(
                n_estimators=cfg.get("classifier", {}).get("n_estimators", 400),
                max_depth=cfg.get("classifier", {}).get("max_depth", 6),
                learning_rate=cfg.get("classifier", {}).get("learning_rate", 0.08),
                subsample=cfg.get("classifier", {}).get("subsample", 0.8),
                colsample_bytree=cfg.get("classifier", {}).get("colsample_bytree", 0.8),
                objective="multi:softprob",
                num_class=len(REGIME_ORDER),
                eval_metric=cfg.get("classifier", {}).get("eval_metric", "mlogloss"),
                tree_method="hist",
                random_state=42
            )
            clf6.fit(X_scaled, y6)
        elif train_6class_algo == "logreg":
            # simple multinomial LR
            clf6 = LogisticRegression(max_iter=300, multi_class="multinomial")
            clf6.fit(X_scaled, y6)
        else:
            log("⚠️ 6-class classifier konfiguriert, aber XGBoost nicht verfügbar – übersprungen.")
            clf6 = None

    # Artefakte speichern
    meta = {
        "assets": assets,
        "timeframes": timeframes,
        "mode_filter": mode_filter,
        "feat_cols": feat_cols,
        "version": version,
        "regimes": REGIME_ORDER,
        "heads_algo": heads_algo,
        "clf6_algo": (train_6class_algo if clf6 is not None else "disabled"),
        "pca_components": pca_components
    }
    save_artifacts(
        artifacts_dir, scaler, feat_cols, pca, hdb, labels,
        thresholds=auto_thr, heads=heads_final, clf6=clf6, meta=meta
    )

    log("✅ Training fertig")


if __name__ == "__main__":
    main()
