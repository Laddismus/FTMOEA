# -*- coding: utf-8 -*-
"""
RegimeService: lädt Scaler, 6-Class-Classifier und Heads.
Gibt pro Zeile:
- regime_6class / conf_6class (QC/Fallback)
- regime_heads / conf_heads    (Direkt aus Heads)
- regime / conf                (Production: Heads + Thresholds aus meta)
"""

import json, numpy as np, pandas as pd
from pathlib import Path
from joblib import load

class RegimeService:
    def __init__(self, artifacts_dir: str):
        self.dir = Path(artifacts_dir)
        self.scaler = load(self.dir / "scaler.joblib")
        self.clf6   = load(self.dir / "clf.joblib")        # 6-Class
        self.feats  = json.loads((self.dir / "feature_cols.json").read_text())
        self.meta   = json.loads((self.dir / "meta.json").read_text())
        self.classes= self.meta.get("classes", list(getattr(self.clf6, "classes_", [])))

        # Heads (optional)
        hdir = self.dir / "heads"
        self.heads = {}
        for name, fn in [("BULL","bull.joblib"),("RANGE","range.joblib"),("VOL","vol.joblib")]:
            p = hdir / fn
            if p.exists():
                self.heads[name] = load(p)

        self.th = (self.meta.get("heads_thresholds") or {"bull":0.5,"range":0.5,"vol":0.5})

    def _xstd(self, row: pd.Series) -> np.ndarray:
        x = pd.DataFrame([[row.get(c, 0.0) for c in self.feats]], columns=self.feats).astype(float)
        return self.scaler.transform(x)[0]

    @staticmethod
    def _combine_heads(b, r, v) -> str:
        if r == 1: return "Range_Volatile" if v==1 else "Range_Quiet"
        side = "Bull" if b==1 else "Bear"
        voln = "Volatile" if v==1 else "Quiet"
        return f"{side}_{voln}"

    def predict(self, row: pd.Series) -> dict:
        x = self._xstd(row).reshape(1, -1)

        # 6-Class (QC / Fallback)
        proba6 = self.clf6.predict_proba(x)[0]
        idx6 = int(np.argmax(proba6))
        regime6 = str(self.classes[idx6]) if self.classes else str(idx6)
        conf6   = float(proba6[idx6])

        # Heads (Production)
        regimeH, confH, final_regime, final_conf = None, None, None, None
        if self.heads:
            pb = float(self.heads["BULL"].predict_proba(x)[0,1])
            pr = float(self.heads["RANGE"].predict_proba(x)[0,1])
            pv = float(self.heads["VOL"].predict_proba(x)[0,1])

            # Heads-Only (0.5) – Diagnose
            b05 = int(pb>=0.5); r05 = int(pr>=0.5); v05 = int(pv>=0.5)
            regimeH = self._combine_heads(b05, r05, v05)
            confH   = float(min(pb if b05 else 1-pb, pr if r05 else 1-pr, pv if v05 else 1-pv))  # konservativ

            # Production thresholds aus meta
            b = int(pb >= float(self.th.get("bull", 0.5)))
            r = int(pr >= float(self.th.get("range",0.5)))
            v = int(pv >= float(self.th.get("vol",  0.5)))
            final_regime = self._combine_heads(b, r, v)
            final_conf = float(min(
                pb if b else 1-pb,
                pr if r else 1-pr,
                pv if v else 1-pv
            ))
        else:
            final_regime, final_conf = regime6, conf6

        # einfache Drift-Heuristik
        z = np.abs(self._xstd(row))
        drift = bool((z > 5).mean() > 0.2)

        return {
            "regime_6class": regime6, "conf_6class": conf6,
            "regime_heads": regimeH,  "conf_heads": confH,
            "regime": final_regime,   "conf": final_conf,   # PRODUCTION OUTPUT
            "drift": drift
        }
