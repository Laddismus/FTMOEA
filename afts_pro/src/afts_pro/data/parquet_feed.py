from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


class MissingColumnsError(ValueError):
    """
    Raised when a parquet file misses required OHLCV columns.
    """


class ParquetFeed:
    """
    Loads OHLCV bars from parquet files.
    """

    REQUIRED_COLUMNS: Sequence[str] = ("timestamp", "open", "high", "low", "close", "volume")

    def __init__(self, data_root: Path) -> None:
        self._data_root = Path(data_root)

    def load(self, symbol: str, folder: str = "final_agg") -> pd.DataFrame:
        filename = symbol if symbol.endswith(".parquet") else f"{symbol}.parquet"
        file_path = self._data_root / folder / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        logger.info("Loading parquet feed from %s", file_path)
        df = pd.read_parquet(file_path)
        if "timestamp" not in df.columns and "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        self._validate_columns(df, file_path)

        df = df.copy()
        timestamp_series = df["timestamp"]
        if pd.api.types.is_numeric_dtype(timestamp_series):
            df["timestamp"] = pd.to_datetime(timestamp_series, unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(timestamp_series, utc=True)
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        return df

    def _validate_columns(self, df: pd.DataFrame, file_path: Path) -> None:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise MissingColumnsError(
                f"Missing columns in {file_path.name}: {', '.join(missing)}"
            )
