from __future__ import annotations

import logging
from collections import deque
import math
from typing import Deque, Optional

from afts_pro.core import MarketState
from afts_pro.features.base_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class CloseReturnCalculator(BaseFeatureCalculator):
    def __init__(self, name: str, lookback: int = 1, **params) -> None:
        super().__init__(name, lookback=lookback, **params)
        self.lookback = max(1, lookback)
        self.closes: Deque[float] = deque(maxlen=self.lookback + 1)

    def update(self, bar: MarketState) -> None:
        self.closes.append(bar.close)

    def current_value(self) -> Optional[float]:
        if len(self.closes) <= self.lookback:
            return None
        past = list(self.closes)[-self.lookback - 1]
        current = self.closes[-1]
        if past == 0:
            return None
        return (current / past) - 1.0


class RollingVolCalculator(BaseFeatureCalculator):
    def __init__(self, name: str, window: int = 20, **params) -> None:
        super().__init__(name, window=window, **params)
        self.window = max(1, window)
        self.returns: Deque[float] = deque(maxlen=self.window)
        self.last_close: Optional[float] = None

    def update(self, bar: MarketState) -> None:
        if self.last_close is not None and self.last_close != 0:
            ret = (bar.close / self.last_close) - 1.0
            self.returns.append(ret)
        self.last_close = bar.close

    def current_value(self) -> Optional[float]:
        if len(self.returns) < 2:
            return None
        mean = sum(self.returns) / len(self.returns)
        var = sum((r - mean) ** 2 for r in self.returns) / (len(self.returns) - 1)
        return var**0.5


class ATRCalculator(BaseFeatureCalculator):
    def __init__(self, name: str, period: int = 14, **params) -> None:
        super().__init__(name, period=period, **params)
        self.period = max(1, period)
        self.prev_close: Optional[float] = None
        self.tr_values: Deque[float] = deque(maxlen=self.period)

    def update(self, bar: MarketState) -> None:
        if self.prev_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - self.prev_close),
                abs(bar.low - self.prev_close),
            )
        self.tr_values.append(tr)
        self.prev_close = bar.close

    def current_value(self) -> Optional[float]:
        if not self.tr_values:
            return None
        return sum(self.tr_values) / len(self.tr_values)


class EMACalculator(BaseFeatureCalculator):
    def __init__(self, name: str, period: int = 14, **params) -> None:
        super().__init__(name, period=period, **params)
        self.period = max(1, period)
        self.alpha = 2 / (self.period + 1)
        self.ema: Optional[float] = None

    def update(self, bar: MarketState) -> None:
        if self.ema is None:
            self.ema = bar.close
        else:
            self.ema = self.alpha * bar.close + (1 - self.alpha) * self.ema

    def current_value(self) -> Optional[float]:
        return self.ema


class RSICalculator(BaseFeatureCalculator):
    def __init__(self, name: str, period: int = 14, **params) -> None:
        super().__init__(name, period=period, **params)
        self.period = max(1, period)
        self.alpha = 1 / float(self.period)
        self.prev_close: Optional[float] = None
        self.gain_ema: Optional[float] = None
        self.loss_ema: Optional[float] = None

    def update(self, bar: MarketState) -> None:
        if self.prev_close is None:
            self.prev_close = bar.close
            return

        change = bar.close - self.prev_close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        if self.gain_ema is None or self.loss_ema is None:
            self.gain_ema = gain
            self.loss_ema = loss
        else:
            self.gain_ema = self.gain_ema + self.alpha * (gain - self.gain_ema)
            self.loss_ema = self.loss_ema + self.alpha * (loss - self.loss_ema)

        self.prev_close = bar.close

    def current_value(self) -> Optional[float]:
        if self.gain_ema is None or self.loss_ema is None:
            return None
        if self.loss_ema == 0:
            return 100.0
        rs = self.gain_ema / self.loss_ema
        return 100.0 - (100.0 / (1.0 + rs))


class VolatilityScoreCalculator(BaseFeatureCalculator):
    def __init__(self, name: str, period: int = 14, **params) -> None:
        super().__init__(name, period=period, **params)
        self.atr_calc = ATRCalculator(name=f"{name}_atr", period=period, **params)
        self.last_close: Optional[float] = None

    def update(self, bar: MarketState) -> None:
        self.last_close = bar.close
        self.atr_calc.update(bar)

    def current_value(self) -> Optional[float]:
        atr = self.atr_calc.current_value()
        if atr is None or self.last_close is None or self.last_close <= 0:
            return None
        return atr / self.last_close


class TrendScoreCalculator(BaseFeatureCalculator):
    def __init__(self, name: str, lookback: int = 20, **params) -> None:
        super().__init__(name, lookback=lookback, **params)
        self.lookback = max(1, lookback)
        self.closes: Deque[float] = deque(maxlen=self.lookback + 1)

    def update(self, bar: MarketState) -> None:
        self.closes.append(bar.close)

    def current_value(self) -> Optional[float]:
        if len(self.closes) <= 1:
            return None
        oldest = self.closes[0]
        latest = self.closes[-1]
        if latest <= 0:
            return None
        n = max(len(self.closes) - 1, 1)
        score = (latest - oldest) / (n * latest)
        return max(min(score, 1.0), -1.0)
