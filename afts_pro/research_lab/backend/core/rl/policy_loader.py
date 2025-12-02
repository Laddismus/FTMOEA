"""Filesystem-based policy loader (stub)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from research_lab.backend.core.rl.models import RLPolicyRef, RLAlgo
from research_lab.backend.core.utils.datetime import ensure_utc_datetime


class RLPolicyLoader:
    """Load RL policies from a policies directory."""

    def __init__(self, policies_dir: Path) -> None:
        self.policies_dir = policies_dir
        self.policies_dir.mkdir(parents=True, exist_ok=True)

    def list_policies(self) -> List[RLPolicyRef]:
        """List policy metadata from *.json files in the policies directory."""

        refs: List[RLPolicyRef] = []
        for file in sorted(self.policies_dir.glob("*.json")):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                data["created_at"] = ensure_utc_datetime(data.get("created_at", "1970-01-01T00:00:00Z"))
                refs.append(RLPolicyRef(**data))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return refs

    def get_policy(self, key: str) -> Optional[RLPolicyRef]:
        """Get policy reference by key."""

        for ref in self.list_policies():
            if ref.key == key:
                return ref
        return None


__all__ = ["RLPolicyLoader"]
