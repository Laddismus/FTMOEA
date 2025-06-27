# Neue Entry-Architektur für institutionelle Logik
# Fokus: Orderflow, Struktur, Liquidity, Confluence Scoring + Logging

class InstitutionalEntrySystem:
    def __init__(self, df_m15, df_h4):
        self.df = df_m15
        self.context_df = df_h4
        self.logs = []
        self.entry_records = []

    def detect_orderblock(self, idx):
        # Platzhalter: echte OB-Erkennung folgt (z. B. nach Break-of-Structure-Candle)
        return True

    def detect_fvg(self, idx):
        if idx < 2: return False
        c1 = self.df.iloc[idx-2]
        c2 = self.df.iloc[idx-1]
        c3 = self.df.iloc[idx]
        return c2["low"] > c1["high"] and c3["low"] > c2["high"]

    def detect_liquidity_sweep(self, idx):
        if idx < 5: return False
        recent_lows = self.df["low"].iloc[idx-5:idx]
        return self.df["low"].iloc[idx] < recent_lows.min()

    def detect_bos(self, idx):
        if idx < 3: return False
        prev_highs = self.df["high"].iloc[idx-3:idx]
        return self.df["high"].iloc[idx] > prev_highs.max()

    def is_volume_cluster(self, idx):
        if idx < 20: return False
        vol = self.df["volume"].iloc[idx-5:idx+1]
        return vol.mean() > self.df["volume"].rolling(20).mean().iloc[idx]

    def get_confluence_score(self, idx):
        score = 0
        criteria = {
            "orderblock": self.detect_orderblock(idx),
            "fvg": self.detect_fvg(idx),
            "liquidity_sweep": self.detect_liquidity_sweep(idx),
            "bos": self.detect_bos(idx),
            "volume_cluster": self.is_volume_cluster(idx)
        }
        for key, result in criteria.items():
            score += int(result)
        return score, criteria

    def evaluate_entries(self, min_score=3):
        entries = []
        for i in range(len(self.df)):
            score, criteria = self.get_confluence_score(i)
            self.logs.append({
                "index": i,
                "time": self.df.index[i],   # ✅ Zeitstempel ergänzen
                "score": score,
                **criteria
            })
            if score >= min_score:
                entry = {
                    "time": self.df.index[i],
                    "score": score,
                    "price": self.df["close"].iloc[i],
                    **criteria
                }
                entries.append(entry)
                self.entry_records.append(entry)
        return entries

    def export_diagnostics(self, filename="results/institutional_entry_log.csv"):
        import pandas as pd
        df_log = pd.DataFrame(self.logs)
        df_log.to_csv(filename, index=False)
        return df_log
