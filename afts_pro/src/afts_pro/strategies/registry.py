from typing import Dict, Optional, Type

from afts_pro.strategies.base import BaseStrategy
from afts_pro.strategies.dummy_ml import DummyMLStrategy
from afts_pro.strategies.orb import OrbStrategy


class StrategyRegistry:
    """
    Maps strategy names to their classes.
    """

    _registry: Dict[str, Type[BaseStrategy]] = {
        "orb": OrbStrategy,
        "dummy_ml": DummyMLStrategy,
    }

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseStrategy]]:
        return cls._registry.get(name)

    @classmethod
    def available(cls) -> Dict[str, Type[BaseStrategy]]:
        return dict(cls._registry)
