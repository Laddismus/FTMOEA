from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from afts_pro.core.benchmark_report import BenchmarkReport


@dataclass
class ModelSelectionCriteria:
    min_profit_factor: float = 1.1
    max_drawdown_pct: float = -10.0
    min_winrate: float = 0.45
    require_ftmo_daily_pass: bool = True
    require_ftmo_overall_pass: bool = True
    min_score: float = 0.0


@dataclass
class ModelSelectionConfig:
    eval_root: str
    agent_type: str
    criteria: ModelSelectionCriteria
    promotion_root: str
    promotion_tag: str
    copy_checkpoint: bool = False
    pointer_filename: str = "CURRENT.txt"
    registry_path: Optional[str] = None


class ModelSelector:
    def __init__(self, cfg: ModelSelectionConfig):
        self.cfg = cfg

    def discover_reports(self) -> List[BenchmarkReport]:
        reports: List[BenchmarkReport] = []
        root = Path(self.cfg.eval_root)
        for path in root.rglob("benchmark_*.json"):
            try:
                data = json.loads(path.read_text())
                rep = BenchmarkReport(
                    kpis=data.get("kpis", {}),
                    ftmo=data.get("ftmo", {}),
                    rl_train=data.get("rl_train", {}),
                    score=data.get("score", 0.0),
                    checkpoint_path=data.get("checkpoint_path", ""),
                    comments=data.get("comments", []),
                )
                reports.append(rep)
            except Exception:
                continue
        return reports

    def filter_reports(self, reports: List[BenchmarkReport]) -> List[BenchmarkReport]:
        crit = self.cfg.criteria
        filtered: List[BenchmarkReport] = []
        for r in reports:
            pf = float(r.kpis.get("profit_factor", 0.0))
            winr = float(r.kpis.get("winrate", 0.0))
            mdd_pct = float(r.kpis.get("mdd_pct", r.kpis.get("max_dd_pct", 0.0)))
            daily_pass = bool(r.ftmo.get("daily_dd_pass", r.ftmo.get("daily_dd", True)))
            overall_pass = bool(r.ftmo.get("overall_dd_pass", r.ftmo.get("overall_dd", True)))
            if pf < crit.min_profit_factor:
                continue
            if mdd_pct < crit.max_drawdown_pct:
                continue
            if winr < crit.min_winrate:
                continue
            if crit.require_ftmo_daily_pass and not daily_pass:
                continue
            if crit.require_ftmo_overall_pass and not overall_pass:
                continue
            if r.score < crit.min_score:
                continue
            filtered.append(r)
        return filtered

    def rank_reports(self, reports: List[BenchmarkReport]) -> List[BenchmarkReport]:
        def _key(r: BenchmarkReport):
            pf = float(r.kpis.get("profit_factor", 0.0))
            mdd_pct = float(r.kpis.get("mdd_pct", r.kpis.get("max_dd_pct", 0.0)))
            return (r.score, pf, -mdd_pct)

        return sorted(reports, key=_key, reverse=True)

    def select_best(self) -> Optional[BenchmarkReport]:
        discovered = self.discover_reports()
        filtered = self.filter_reports(discovered)
        ranked = self.rank_reports(filtered)
        return ranked[0] if ranked else None

    def promote(self, best: BenchmarkReport, dry_run: bool = False) -> Path:
        target_root = Path(self.cfg.promotion_root) / self.cfg.promotion_tag / self.cfg.agent_type
        target_root.mkdir(parents=True, exist_ok=True)
        pointer_path = target_root / self.cfg.pointer_filename
        meta_path = target_root / "selection_info.json"
        if dry_run:
            return target_root
        if self.cfg.copy_checkpoint:
            src = Path(best.checkpoint_path)
            if src.is_dir():
                dest = target_root / src.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, target_root / src.name)
        else:
            pointer_path.write_text(str(Path(best.checkpoint_path).resolve()), encoding="utf-8")
        meta = {
            "score": best.score,
            "kpis": best.kpis,
            "ftmo": best.ftmo,
            "rl_train": best.rl_train,
            "checkpoint": best.checkpoint_path,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if self.cfg.registry_path:
            self._append_registry_record(meta)
        return target_root

    def _append_registry_record(self, record: dict) -> None:
        reg_path = Path(self.cfg.registry_path)
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if reg_path.exists():
            try:
                existing = json.loads(reg_path.read_text())
            except Exception:
                existing = []
        existing.append(record)
        reg_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
