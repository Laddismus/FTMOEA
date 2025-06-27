import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class TrendFilter:
    def __init__(self, df):
        self.df = df.copy()
        self.df["ema"] = self.df["close"].ewm(span=50).mean()

    def compute_slope(self, series):
        y = series.values.reshape(-1, 1)
        x = np.arange(len(series)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        return model.coef_[0][0]

    def label_trend(self, df=None, window=10, slope_threshold=0.0001, divergence_threshold=0.0008):
        if df is None:
            df = self.df.copy()
        df = df.copy()
        df["ema"] = df["close"].ewm(span=50).mean()

        trends = []
        for i in range(len(df)):
            if i < window:
                trends.append("undefined")
                continue

            ema_slice = df["ema"].iloc[i - window:i]
            slope = self.compute_slope(ema_slice)
            price = df["close"].iloc[i]
            ema_now = df["ema"].iloc[i]
            divergence = abs(price - ema_now)

            if slope > slope_threshold and divergence > divergence_threshold:
                trends.append("bullish")
            elif slope < -slope_threshold and divergence > divergence_threshold:
                trends.append("bearish")
            else:
                trends.append("sideways")

        df = df.iloc[window:].copy()
        df["trend"] = trends[window:]
        return df[["close", "ema", "trend"]]

    def get_trend_at(self, current_time, window=10, slope_threshold=0.0001, divergence_threshold=0.0005):
        df = self.df
        idx = df.index.get_indexer([current_time], method='pad')[0]

        if idx < window:
            return "undefined"

        ema_slice = df["ema"].iloc[idx - window:idx]
        slope = self.compute_slope(ema_slice)
        price = df["close"].iloc[idx]
        ema_now = df["ema"].iloc[idx]
        divergence = abs(price - ema_now)

        if slope > slope_threshold and divergence > divergence_threshold:
            return "bullish"
        elif slope < -slope_threshold and divergence > divergence_threshold:
            return "bearish"
        else:
            return "sideways"
