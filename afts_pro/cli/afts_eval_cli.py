from __future__ import annotations

import argparse
import logging
from pathlib import Path
import json

from afts_pro.core.eval_controller import EvalConfig, EvalController

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="AFTS-PRO EVAL pipeline CLI")
    parser.add_argument("--profile", default="configs/modes/eval.yaml", help="Eval config path.")
    parser.add_argument("--checkpoint", help="Checkpoint path for evaluation.")
    parser.add_argument("--multi", help="JSON file containing list of checkpoint paths.")
    parser.add_argument("--no-html", action="store_true", help="Disable HTML report generation for this run.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg = EvalConfig.from_yaml(args.profile)
    if args.no_html:
        cfg.output["save_html"] = False
    controller = EvalController(cfg)

    if args.multi:
        checkpoints = json.loads(Path(args.multi).read_text())
        comparison = controller.run_multi_eval(checkpoints)
        logger.info("Best checkpoint: %s", comparison.best_checkpoint)
    else:
        report = controller.run_eval(checkpoint_path=args.checkpoint)
        logger.info("Eval done | checkpoint=%s | score=%.4f", report.checkpoint_path, report.score)


if __name__ == "__main__":
    main()
