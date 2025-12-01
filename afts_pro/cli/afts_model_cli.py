from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from afts_pro.core.model_selection import ModelSelectionConfig, ModelSelectionCriteria, ModelSelector

logger = logging.getLogger(__name__)


def _load_profile(name: str, path: str) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    profiles = data.get("model_selection_profiles", {})
    if name not in profiles:
        raise ValueError(f"Unknown model selection profile: {name}")
    return profiles[name]


def _build_selector(profile: dict) -> ModelSelector:
    crit_data = profile.get("criteria", {})
    criteria = ModelSelectionCriteria(
        min_profit_factor=crit_data.get("min_profit_factor", 1.1),
        max_drawdown_pct=crit_data.get("max_drawdown_pct", -10.0),
        min_winrate=crit_data.get("min_winrate", 0.45),
        require_ftmo_daily_pass=crit_data.get("require_ftmo_daily_pass", True),
        require_ftmo_overall_pass=crit_data.get("require_ftmo_overall_pass", True),
        min_score=crit_data.get("min_score", 0.0),
    )
    cfg = ModelSelectionConfig(
        eval_root=profile["eval_root"],
        agent_type=profile["agent_type"],
        criteria=criteria,
        promotion_root=profile["promotion_root"],
        promotion_tag=profile["promotion_tag"],
        copy_checkpoint=profile.get("copy_checkpoint", False),
        pointer_filename=profile.get("pointer_filename", "CURRENT.txt"),
        registry_path=profile.get("registry_path"),
    )
    return ModelSelector(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="AFTS Model Selection & Promotion CLI")
    sub = parser.add_subparsers(dest="cmd")
    parser_list = sub.add_parser("list-profiles", help="List available model selection profiles.")
    parser_select = sub.add_parser("select", help="Select best model for a profile.")
    parser_select.add_argument("--profile", required=True)
    parser_select.add_argument("--config", default="configs/model_selection.yaml")
    parser_select.add_argument("--dry-run", action="store_true")
    parser_promote = sub.add_parser("promote", help="Select and promote best model.")
    parser_promote.add_argument("--profile", required=True)
    parser_promote.add_argument("--config", default="configs/model_selection.yaml")
    parser_promote.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.cmd == "list-profiles":
        data = yaml.safe_load(Path("configs/model_selection.yaml").read_text())
        for name in data.get("model_selection_profiles", {}).keys():
            print(name)
        return

    if args.cmd in ("select", "promote"):
        profile_data = _load_profile(args.profile, args.config)
        selector = _build_selector(profile_data)
        best = selector.select_best()
        if best is None:
            logger.error("No model matched criteria for profile %s", args.profile)
            raise SystemExit(1)
        logger.info("Best model: %s | score=%.4f | pf=%.3f | mdd=%.3f | winrate=%.3f", best.checkpoint_path, best.score, best.kpis.get("profit_factor", 0.0), best.kpis.get("mdd_pct", best.kpis.get("max_dd_pct", 0.0)), best.kpis.get("winrate", 0.0))
        if args.cmd == "promote":
            target = selector.promote(best, dry_run=args.dry_run)
            logger.info("Promotion %s | target=%s", "DRY-RUN" if args.dry_run else "DONE", target)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
