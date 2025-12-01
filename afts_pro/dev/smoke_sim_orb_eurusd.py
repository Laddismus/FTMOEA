from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from afts_pro.core.e2e_runner import E2ESimConfig, run_e2e_sim  # noqa: E402

CONFIG_PATH = Path("configs/sim_smoke_orb_eurusd.yaml")


def load_cfg(path: Path) -> dict:
    if path.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(path.read_text()) or {}
    return json.loads(path.read_text())


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if not CONFIG_PATH.exists():
        logging.error("[SMOKE-SIM] Config not found: %s", CONFIG_PATH)
        sys.exit(1)
    cfg_data = load_cfg(CONFIG_PATH)
    e2e_cfg_path = Path("artifacts/dev_smoke_sim/e2e_cfg.json")
    e2e_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    e2e_cfg_path.write_text(json.dumps(cfg_data, indent=2))
    logging.info("[SMOKE-SIM] Starting E2E SIM | cfg=%s", CONFIG_PATH)
    result = run_e2e_sim(E2ESimConfig(config_path=str(e2e_cfg_path)))
    logging.info(
        "[SMOKE-SIM] Done | trades=%d | equity_start=%.2f | equity_end=%.2f | run_dir=%s",
        result.num_trades,
        result.equity_start,
        result.equity_end,
        result.run_dir,
    )
    if not result.files_present.get("trades") or not result.files_present.get("equity"):
        logging.error("[SMOKE-SIM] Missing expected artifacts: %s", result.files_present)
        sys.exit(1)


if __name__ == "__main__":
    main()
