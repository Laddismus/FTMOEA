from __future__ import annotations

import abc
from typing import Optional

from afts_pro.core import MarketState
from afts_pro.features.state import ExtrasSnapshot


class BaseFeatureCalculator(abc.ABC):
    def __init__(self, name: str, **params) -> None:
        self.name = name
        self.params = params

    @abc.abstractmethod
    def update(self, bar: MarketState, extras: Optional[ExtrasSnapshot] = None) -> None:
        """
        Consume the latest bar and update internal state. Must be lookahead-safe.
        """

    @abc.abstractmethod
    def current_value(self):
        """
        Return current feature value for the latest bar. May return None if insufficient history.
        """
