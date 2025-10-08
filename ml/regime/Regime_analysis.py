# -*- coding: utf-8 -*-
"""
Regime Analysis
- Lädt Artefakte (scaler, pca, heads, optional clf6, thresholds)
- Lädt Feature-Parquet (z. B. EURUSD_5m_BACKTEST.parquet)
- Macht Batch-Predictions (Heads→6Class, optional 6Class direct)
- Statistik: Counts, Transition-Matrix, Drift-Rate (Placeholder), QC-Consistency
- Export CSV + optional Markdown-Report

Benötigt: kits_paths.py im Projekt-Root.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

# Artefakt-IO
import joblib

from kits_paths import p

REGIME_ORDER = ["Bear_Quiet", "Bear_Volatile", "Bull_Quiet", "Bull_Volatile", "Range_Quiet", "Range_Volatile"]

def log(msg: str):
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def ensure_dt64_utc(df, col="timestamp"):
    if col not in df.columns:
        raise KeyError(f"Spalte '{col}' fehlt.")
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], utc=True)
    elif getattr(df[col].dt, "tz", None) is None:
        df[col] = df[col].dt.tz_localize("UTC")
    return df

def load_artifacts(art_dir: Path):
    scaler = joblib.load(art_dir / "scaler.joblib")
    pca    = joblib.load(art_dir / "pca.joblib")
    thresholds = json.loads((art_dir / "thresholds.json").read_text())
    feat_cols  = json.loads((art_dir / "feat_cols.json").read_text())
    meta = json.loads((art_dir / "meta.json").read_text())

    heads = {}
    for k in ["bull","range","vol"]:
        fp = art_dir / f"head_{k}.joblib"
        if fp.exists():
            heads[k] = joblib.load(fp)

    clf6 = None
    fp6 = art_dir / "clf6.joblib"
    if fp6.exists():
        try:
            clf6 = joblib.load(fp6)
        except Exception:
            clf6 = None

    return {"scaler": scaler, "pca": pca, "thresholds": thresholds, "feat_cols": feat_cols,
            "heads": heads, "clf6": clf6, "meta": meta}

def predict_heads_proba(heads: dict, X: np.ndarray) -> pd.DataFrame:
    out = {}
    for k, h in heads.items():
        proba = h.predict_proba(X)
        out[k] = proba[:, list(h.classes_).index(1)]
    return pd.DataFrame(out)

def heads_to_6class(proba_heads: pd.DataFrame, thresholds: dict) -> tuple[np.ndarray, np.ndarray]:
    bull_t = thresholds.get("bull", 0.6)
    range_t= thresholds.get("range", 0.5)
    vol_t  = thresholds.get("vol", 0.55)

    bull = (proba_heads["bull"]  >= bull_t).astype(int).values
    ran  = (proba_heads["range"] >= range_t).astype(int).values
    vol  = (proba_heads["vol"]   >= vol_t).astype(int).values

    labels = []
    conf   = []
    for b, r, v, pb, pr, pv in zip(bull, ran, vol,
                                   proba_heads["bull"].values,
                                   proba_heads["range"].values,
                                   proba_heads["vol"].values):
        if r == 1:
            labels.append("Range_Volatile" if v else "Range_Quiet")
            conf.append(max(pr, pv))
        else:
            if b == 1:
                labels.append("Bull_Volatile" if v else "Bull_Quiet")
                conf.append(max(pb, pv))
            else:
                labels.append("Bear_Volatile" if v else "Bear_Quiet")
                conf.append(max(1.0 - pb, pv))
    return np.array(labels), np.array(conf, dtype=float)

def transition_matrix(series: pd.Series, classes: list[str]) -> pd.DataFrame:
    idx = pd.Categorical(series, categories=classes, ordered=False).codes
    mat = np.zeros((len(classes), len(classes)), dtype=float)
    for i in range(len(idx) - 1):
        if idx[i] < 0 or idx[i+1] < 0:
            continue
        mat[idx[i], idx[i+1]] += 1.0
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums
    return pd.DataFrame(mat, index=classes, columns=classes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=str, default=None, help="Pfad zu Artefakten (Standard: models_root/version)")
    parser.add_argument("--features", type=str, default=None, help="Feature-Parquet, überschreibt asset/tf/mode")
    parser.add_argument("--asset", type=str, default="EURUSD")
    parser.add_argument("--tf", type=str, default="5m")
    parser.add_argument("--mode", type=str, default="BACKTEST")
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--export_csv", type=str, default=None)
    parser.add_argument("--show_qc", action="store_true", help="zusätzliche Konsistenz-Tabellen ausgeben")
    args = parser.parse_args()

    # Artefakte
    meta_dir = p("models_root")
    # Default version ist aus meta.json, aber wir probieren v1, falls nicht angegeben:
    version_dir = None
    if args.artifacts:
        version_dir = Path(args.artifacts)
    else:
        # rate: v1
        if (meta_dir / "v1" / "meta.json").exists():
            version_dir = meta_dir / "v1"
        else:
            # fallback direkt root
            version_dir = meta_dir

    arts = load_artifacts(version_dir)

    # Features
    if args.features:
        fpath = Path(args.features)
    else:
        fpath = p("features_dir") / f"{args.asset}_{args.tf}_{args.mode}.parquet"

    print("".join([
        "\n=======================\n",
        "Regime Analysis — Setup\n",
        "=======================\n",
        f"Artifacts: {version_dir}\n",
        f"Features/Labeled: {fpath}\n",
    ]))

    df = pd.read_parquet(fpath)
    df = ensure_dt64_utc(df, "timestamp").sort_values("timestamp")
    if args.rows and args.rows > 0:
        df = df.tail(args.rows).copy()
    print(f"Rows used: {len(df):,}\n")

    feat_cols = arts["feat_cols"]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Features fehlen im Parquet: {missing}")

    X = df[feat_cols].values
    Xs = arts["scaler"].transform(X)
    Xp = arts["pca"].transform(Xs)

    # Heads → 6Class
    proba_heads = predict_heads_proba(arts["heads"], Xs)
    y_hat, conf = heads_to_6class(proba_heads, arts["thresholds"])

    # Optional: 6-Klassen-Modell
    y6 = None
    if arts["clf6"] is not None:
        try:
            proba6 = arts["clf6"].predict_proba(Xs)
            # max-Klasse
            idx = np.argmax(proba6, axis=1)
            y6 = np.array(REGIME_ORDER)[idx]
            print("[QC] 6-class clf verfügbar.")
        except Exception:
            y6 = None

    # Ausgabe: letzte 5
    print("1) Sample Predictions (letzte 5 Candles)\n----------------------------------------")
    tail_idx = df.index[-5:]
    for i in tail_idx:
        rec = {
            "regime_6class": (y6[df.index.get_loc(i)] if y6 is not None else None),
            "conf_6class": (float(np.max(proba6[df.index.get_loc(i)])) if arts["clf6"] is not None else None),
            "regime_heads": str(y_hat[df.index.get_loc(i)]),
            "conf_heads": float(conf[df.index.get_loc(i)]),
        }
        # Final regime == heads (wir folgen Heads-Linie)
        rec["regime"] = rec["regime_heads"]
        rec["conf"]   = rec["conf_heads"]
        rec["drift"]  = False  # Platzhalter – Drift-Detector kann später ergänzt werden
        print(f"{df.loc[i, 'timestamp']} {rec}")
    print()

    # Batch-Result
    out = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "regime": y_hat,
        "conf": conf,
    }, index=df.index)

    # Statistiken
    print("\n2) Batch Predictions (Regime & Confidence)\n------------------------------------------\n")
    print("Regime Counts:")
    print(out["regime"].value_counts().sort_index(), "\n")
    print(f"Durchschnittliche Confidence: {out['conf'].mean():.3f}\n")

    # ATR-Profil pro Regime (nur wenn verfügbar)
    if "atr_pips_14" in df.columns:
        tmp = out.join(df[["atr_pips_14"]], how="left")
        print("Mean ATR (pips) je Regime:")
        print(tmp.groupby("regime")["atr_pips_14"].mean().round(2), "\n")

    # Transition-Matrix
    print("3) Transition-Matrix (Markov-ähnlich)\n-------------------------------------")
    tm = transition_matrix(out["regime"], REGIME_ORDER)
    print(tm, "\n")

    # Drift-Rate Placeholder (hier: Anteil conf < global score → flagged)
    score_thr = arts["thresholds"].get("score", 0.62)
    drift_flag = (out["conf"] < score_thr).astype(int)
    drift_rate = drift_flag.mean()
    print("4) Drift-Rate\n-------------")
    print(f"Drift-Rate (Anteil drift=True): {drift_rate:.3f}\n")

    # Heads vs 6-Class Konsistenz
    print("5) Heads vs 6-Class (Konsistenz)\n--------------------------------")
    if (arts["clf6"] is not None) and (y6 is not None):
        # Kreuztabelle
        cross = pd.crosstab(pd.Series(out["regime"], name="regime"),
                            pd.Series(y6, name="regime_6class"),
                            normalize="index").reindex(index=REGIME_ORDER, columns=REGIME_ORDER, fill_value=0.0)
        print(cross)
        row_acc = cross.max(axis=1).mean()
        print(f"Ø Zeilenmax: {row_acc:.3f}\n")
    else:
        print("6-class Klassifikator nicht vorhanden – QC nur Heads-basiert.\n")

    # Export CSV
    export_path = None
    if args.export_csv:
        export_path = Path(args.export_csv)
    else:
        export_path = p("reports_dir") / f"regime_monitor_{args.asset}_{args.tf}.csv"

    export_df = out.join(df[["atr_pips_14"]] if "atr_pips_14" in df.columns else pd.DataFrame(index=df.index))
    export_df.to_csv(export_path, index=False)
    print(f"Exportiert: {export_path} ({len(export_df):,} Zeilen)")

if __name__ == "__main__":
    main()
