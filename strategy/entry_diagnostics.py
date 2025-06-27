import pandas as pd
import numpy as np

class EntryDiagnostics:
    def __init__(self, df_m15, trade_log):
        self.df = df_m15.copy()
        self.trades = trade_log.copy()
        self.results = []

    def analyze(self):
        for _, trade in self.trades.iterrows():
            entry_time = pd.to_datetime(trade["entry_time"])
            result = trade["result"]
            entry_price = trade["entry_price"]

            if entry_time not in self.df.index:
                continue

            # Hole Candle
            candle = self.df.loc[entry_time]
            high = candle["high"]
            low = candle["low"]
            open_ = candle["open"]
            close = candle["close"]
            volume = candle.get("volume", np.nan)

            # Kerzenstruktur
            body = abs(close - open_)
            range_ = high - low
            wick_ratio = (range_ - body) / range_ if range_ != 0 else 0
            body_strength = body / range_ if range_ != 0 else 0
            is_doji = body_strength < 0.2

            # Momentum (letzte 3 Closes)
            idx = self.df.index.get_loc(entry_time)
            if idx < 3:
                momentum_3 = np.nan
            else:
                momentum_3 = self.df["close"].iloc[idx] - self.df["close"].iloc[idx - 3]

            # Swing-Abstand
            window = self.df.iloc[max(0, idx - 30):idx]
            swing_low = window["low"].min() if not window.empty else np.nan
            swing_high = window["high"].max() if not window.empty else np.nan
            dist_to_swing_low = abs(entry_price - swing_low)
            dist_to_swing_high = abs(entry_price - swing_high)

            # Candle vs ATR (Volatilitätsfaktor)
            atr = (self.df["high"] - self.df["low"]).rolling(window=14).mean()
            atr_value = atr.iloc[idx] if idx < len(atr) else np.nan
            candle_vs_atr = range_ / atr_value if atr_value and atr_value != 0 else np.nan

            # Ergebnis speichern
            self.results.append({
                "entry_time": entry_time,
                "result": result,
                "wick_ratio": wick_ratio,
                "body_strength": body_strength,
                "is_doji": is_doji,
                "momentum_3": momentum_3,
                "dist_to_swing_low": dist_to_swing_low,
                "dist_to_swing_high": dist_to_swing_high,
                "candle_vs_atr": candle_vs_atr,
                "volume": volume
            })

        return pd.DataFrame(self.results)

# Beispiel:
# diag = EntryDiagnostics(df_m15, df_trade_log)
# result_df = diag.analyze()
# result_df.to_csv("results/entry_diagnostics.csv", index=False) # optional speichern
