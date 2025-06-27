from strategy.trend_filter import TrendFilter  # wichtig: Modul muss existieren

class EntryLogic:
    def __init__(self, df_m15, df_h4):
        self.df = df_m15
        self.trend_df = df_h4
        self.trend_filter = TrendFilter(df_h4)
        self.logs = {
            "trend": {"bullish": 0, "bearish": 0, "sideways": 0},
            "checked_setups": 0,
            "breakouts": 0,
            "retests": 0,
            "rejections": 0,
            "entries": 0,
        }

    def find_recent_swing_high_low(self, df_slice, left=2, right=2):
        highs = df_slice["high"].tolist()
        lows = df_slice["low"].tolist()
        swing_high = None
        swing_low = None

        for i in range(left, len(highs) - right):
            if all(highs[i] > highs[i - j] for j in range(1, left + 1)) and \
               all(highs[i] > highs[i + j] for j in range(1, right + 1)):
                swing_high = highs[i]

            if all(lows[i] < lows[i - j] for j in range(1, left + 1)) and \
               all(lows[i] < lows[i + j] for j in range(1, right + 1)):
                swing_low = lows[i]

        return swing_high, swing_low

    def check_entry_signal(self, current_idx):
        if current_idx < 30:
            return False

        df_slice = self.df.iloc[current_idx - 30:current_idx]
        current = self.df.iloc[current_idx]
        current_time = self.df.index[current_idx]

        trend = self.trend_filter.get_trend_at(current_time, window=10, slope_threshold=0.0001, divergence_threshold=0.0008)

        if trend in self.logs["trend"]:
            self.logs["trend"][trend] += 1
        else:
            # Optional: initialisieren, wenn du undefined mitzählen willst
            self.logs["trend"]["undefined"] = self.logs["trend"].get("undefined", 0) + 1


        if trend == "sideways":
            return False

        self.logs["checked_setups"] += 1

        swing_high, swing_low = self.find_recent_swing_high_low(df_slice)

        if trend == "bullish" and swing_high and current["close"] > swing_high:
            self.logs["breakouts"] += 1
            if current["low"] <= swing_high:
                self.logs["retests"] += 1
                if current["close"] > current["open"]:
                    self.logs["rejections"] += 1
                    self.logs["entries"] += 1
                    return True

        elif trend == "bearish" and swing_low and current["close"] < swing_low:
            self.logs["breakouts"] += 1
            if current["high"] >= swing_low:
                self.logs["retests"] += 1
                if current["close"] < current["open"]:
                    self.logs["rejections"] += 1
                    self.logs["entries"] += 1
                    return True

        return False

    def get_logs(self):
        return self.logs
