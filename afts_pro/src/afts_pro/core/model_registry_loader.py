from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ProductionModelRef:
    agent_type: str
    tag: str
    checkpoint_path: Path
    selection_meta: dict
    pointer_file: Path
    root_dir: Path


@dataclass
class ModelRegistryConfig:
    promotion_root: str
    registry_path: Optional[str] = None


class ModelRegistryLoader:
    def __init__(self, cfg: ModelRegistryConfig):
        self.cfg = cfg
        self.root = Path(cfg.promotion_root)

    def has_production_model(self, tag: str) -> bool:
        tag_dir = self.root / tag
        return (tag_dir / "CURRENT.txt").exists()

    def load_production_ref(self, tag: str, agent_type: str) -> ProductionModelRef:
        tag_dir = self.root / tag
        pointer = tag_dir / "CURRENT.txt"
        if not pointer.exists():
            raise FileNotFoundError(f"Production pointer not found for tag {tag} at {pointer}")
        checkpoint_path = Path(pointer.read_text().strip())
        if not checkpoint_path.exists():
            # try relative to tag_dir
            rel = (tag_dir / checkpoint_path).resolve()
            if rel.exists():
                checkpoint_path = rel
            else:
                raise FileNotFoundError(f"Checkpoint path in pointer missing: {checkpoint_path}")
        meta_path = tag_dir / "selection_info.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        return ProductionModelRef(
            agent_type=agent_type,
            tag=tag,
            checkpoint_path=checkpoint_path,
            selection_meta=meta,
            pointer_file=pointer,
            root_dir=tag_dir,
        )

    def load_registry(self) -> list[dict]:
        if not self.cfg.registry_path:
            return []
        reg = Path(self.cfg.registry_path)
        if not reg.exists():
            return []
        try:
            return json.loads(reg.read_text())
        except Exception:
            return []
