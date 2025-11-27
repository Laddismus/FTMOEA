from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from afts_pro.lab.models import LabResult

logger = logging.getLogger(__name__)


def build_kpi_matrix(results: List[LabResult], metrics: List[str]) -> pd.DataFrame:
    """
    Build a KPI matrix DataFrame from LabResult objects.
    """
    rows = []
    for res in results:
        row = {}
        row.update(res.params)
        for metric in metrics:
            row[metric] = res.metrics.get(metric)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def save_kpi_matrix(df, path: Path, fmt: str = "parquet") -> None:
    """
    Save KPI matrix to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    logger.info("Saved KPI matrix to %s", path)
