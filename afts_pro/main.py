from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import typer

# Ensure src/ is on the import path when running as a script.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from afts_pro.core import Mode, ModeDispatcher
from afts_pro.engine import start as engine_start
from afts_pro.utils.logging import configure_logging, setup_logging
from afts_pro.config import (
    global_config_summary,
    list_profile_paths,
    load_global_config_from_profile,
    run_all_validations,
)

try:
    import uvloop
except ImportError:  # pragma: no cover - optional dependency
    uvloop = None


app = typer.Typer(no_args_is_help=True, add_completion=False)
config_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Config validation and dump utilities.")
logger = logging.getLogger(__name__)


async def _start_mode(mode: Mode, profile_path: str) -> None:
    dispatcher = ModeDispatcher(lambda m: engine_start(m, profile_path=profile_path))
    await dispatcher.dispatch(mode)


def _resolve_profile_selection(profile: str, profile_path: Optional[str]) -> Tuple[str, Path]:
    if profile_path:
        resolved_profile = Path(profile_path).resolve()
        profile_name = resolved_profile.stem
        return profile_name, resolved_profile

    available = list_profile_paths(str(ROOT_DIR / "configs" / "profiles"))
    resolved = available.get(profile)
    if resolved is None:
        logger.error("PROFILE_ERROR | unknown profile=%s | available=%s", profile, list(available.keys()))
        raise typer.Exit(code=1)
    resolved_profile = Path(resolved).resolve()
    profile_name = profile
    return profile_name, resolved_profile


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    mode: Mode = typer.Option(
        Mode.SIM,
        "--mode",
        "-m",
        case_sensitive=False,
        help="Execution mode: train, sim, live. Hot-reload available in sim/train when enabled in environment config.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        case_sensitive=False,
        help="Logging level (INFO, DEBUG, WARNING, ERROR).",
    ),
    profile: str = typer.Option(
        "sim",
        "--profile",
        "-p",
        help="Name of config profile (e.g. sim, ftmo, apex, equity).",
    ),
    profile_path: str = typer.Option(
        None,
        "--profile-path",
        help="Explicit path to a profile YAML (overrides --profile).",
    ),
) -> None:
    if ctx.invoked_subcommand:
        return

    setup_logging(level=log_level)
    if uvloop is not None:
        uvloop.install()
        logger.debug("uvloop event loop policy installed")

    # Resolve profile path
    profile_name, resolved_profile = _resolve_profile_selection(profile, profile_path)

    logger.info("PROFILE_SELECTED | name=%s | path=%s", profile_name, resolved_profile)
    logger.info("Starting AFTS-PRO in mode=%s", mode.value)
    asyncio.run(_start_mode(mode, str(resolved_profile)))


@config_app.command("validate")
def config_validate(
    profile: str = typer.Option("sim", "--profile", "-p", help="Name of config profile."),
    profile_path: str = typer.Option(None, "--profile-path", help="Explicit path to a profile YAML."),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level."),
) -> None:
    setup_logging(level=log_level)
    profile_name, resolved_profile = _resolve_profile_selection(profile, profile_path)
    logger.info("PROFILE_SELECTED | name=%s | path=%s", profile_name, resolved_profile)
    global_config = load_global_config_from_profile(str(resolved_profile))
    ok, messages = run_all_validations(global_config, ROOT_DIR / "data" / "final_agg")
    summary = global_config_summary(global_config)
    logger.info("CONFIG SUMMARY | %s", summary)
    for msg in messages:
        if msg.startswith("ERROR"):
            logger.error(msg)
        else:
            logger.warning(msg)
    if not ok:
        raise typer.Exit(code=1)


@config_app.command("dump")
def config_dump(
    profile: str = typer.Option("sim", "--profile", "-p", help="Name of config profile."),
    profile_path: str = typer.Option(None, "--profile-path", help="Explicit path to a profile YAML."),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level."),
) -> None:
    setup_logging(level=log_level)
    profile_name, resolved_profile = _resolve_profile_selection(profile, profile_path)
    logger.info("PROFILE_SELECTED | name=%s | path=%s", profile_name, resolved_profile)
    global_config = load_global_config_from_profile(str(resolved_profile))
    summary = global_config_summary(global_config)
    if format.lower() == "json":
        typer.echo(json.dumps(summary, indent=2))
    else:
        lines = [
            f"Profile: {profile_name}",
            f"Environment: mode={summary['environment']['mode']} timezone={summary['environment']['timezone']}",
            f"Risk Policy: {summary['risk_policy']}",
            f"Behaviour Enabled: {summary['behaviour_enabled']}",
            f"Strategies: {', '.join(summary['strategies'])}",
            f"Assets: {', '.join(summary['assets'])}",
            f"Features Enabled: {summary.get('features_enabled')}, Model Features Enabled: {summary.get('model_features_enabled')}, Scaling={summary.get('model_scaling_type')}",
        ]
        typer.echo("\n".join(lines))


app.add_typer(config_app, name="config")


if __name__ == "__main__":
    app()
