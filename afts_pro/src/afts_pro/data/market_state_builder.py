from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Optional

from afts_pro.core import MarketState
from afts_pro.data.parquet_feed import ParquetFeed


class MarketStateBuilder:
    """
    Creates MarketState instances from parquet feed rows.
    """

    def __init__(self, feed: ParquetFeed) -> None:
        self._feed = feed

    def iter_market_states(self, symbol: str, folder: str = "final_agg") -> Iterator[MarketState]:
        df = self._feed.load(symbol, folder=folder)
        base_symbol = self._normalize_symbol(symbol)

        for row in df.itertuples(index=False):
            row_symbol: Optional[str] = getattr(row, "symbol", None)
            yield MarketState(
                timestamp=row.timestamp.to_pydatetime(),
                symbol=row_symbol or base_symbol,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Accept both full filenames and bare symbols.
        """
        symbol_path = Path(symbol)
        return symbol_path.stem if symbol_path.suffix else symbol
