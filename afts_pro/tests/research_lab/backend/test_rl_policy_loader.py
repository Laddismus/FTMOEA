from datetime import datetime, timezone
from pathlib import Path
import json

from research_lab.backend.core.rl.models import RLAlgo
from research_lab.backend.core.rl.policy_loader import RLPolicyLoader


def test_policy_loader_lists_and_gets(tmp_path: Path) -> None:
    meta = {
        "key": "policy1",
        "algo": RLAlgo.SAC.value,
        "path": "policy1.bin",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {"note": "demo"},
    }
    (tmp_path / "policy1.json").write_text(json.dumps(meta), encoding="utf-8")
    loader = RLPolicyLoader(tmp_path)

    policies = loader.list_policies()
    assert len(policies) == 1
    assert policies[0].key == "policy1"

    fetched = loader.get_policy("policy1")
    assert fetched is not None
    assert fetched.algo == RLAlgo.SAC
