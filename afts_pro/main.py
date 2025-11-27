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
from afts_pro.data import ExtrasLoader
from afts_pro.features import FeatureEngine
from afts_pro.data import ParquetFeed, MarketStateBuilder

try:
    import uvloop
except ImportError:  # pragma: no cover - optional dependency
    uvloop = None


app = typer.Typer(no_args_is_help=True, add_completion=False)
config_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Config validation and dump utilities.")
extras_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Extras utilities.")
runs_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Run history utilities.")
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
            f"Extras Enabled: {summary.get('extras_enabled')} | Datasets: {summary.get('extras_datasets')}",
            f"RunLogger Enabled: {summary.get('runlogger_enabled')} | BaseDir: {summary.get('runlogger_base_dir')}",
        ]
        typer.echo("\n".join(lines))


@extras_app.command("check")
def extras_check(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Symbol to check extras for."),
    profile: str = typer.Option("sim", "--profile", "-p", help="Name of config profile."),
    profile_path: str = typer.Option(None, "--profile-path", help="Explicit path to a profile YAML."),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level."),
) -> None:
    setup_logging(level=log_level)
    profile_name, resolved_profile = _resolve_profile_selection(profile, profile_path)
    logger.info("PROFILE_SELECTED | name=%s | path=%s", profile_name, resolved_profile)
    global_config = load_global_config_from_profile(str(resolved_profile))
    if not global_config.extras.enabled:
        logger.info("Extras loader disabled in config; nothing to check.")
        return
    loader = ExtrasLoader(global_config.extras)
    series_map = loader.load_for_symbol(symbol)
    logger.info("EXTRAS_CHECK | symbol=%s | datasets_loaded=%s", symbol, list(series_map.keys()))


@extras_app.command("preview")
def extras_preview(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Symbol to preview extras for."),
    bars: int = typer.Option(5, "--bars", "-b", help="Number of bars to preview."),
    profile: str = typer.Option("sim", "--profile", "-p", help="Name of config profile."),
    profile_path: str = typer.Option(None, "--profile-path", help="Explicit path to a profile YAML."),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level."),
) -> None:
    setup_logging(level=log_level)
    profile_name, resolved_profile = _resolve_profile_selection(profile, profile_path)
    logger.info("PROFILE_SELECTED | name=%s | path=%s", profile_name, resolved_profile)
    global_config = load_global_config_from_profile(str(resolved_profile))
    if not global_config.extras.enabled:
        logger.info("EXTRAS_PREVIEW | extras disabled in config.")
        return
    extras_loader = ExtrasLoader(global_config.extras)
    extras_map = extras_loader.load_for_symbol(symbol)
    if not extras_map:
        logger.info("EXTRAS_PREVIEW | no extras available for symbol=%s", symbol)
        return
    feature_engine = FeatureEngine(global_config.features)
    feature_engine.attach_extras(extras_map)
    feed = ParquetFeed(ROOT_DIR / "data")
    builder = MarketStateBuilder(feed)
    ms_iter = builder.iter_market_states(symbol=symbol, folder="final_agg")
    for idx, ms in enumerate(ms_iter):
        if idx >= bars:
            break
        bundle = feature_engine.update(ms)
        logger.info(
            "EXTRAS_PREVIEW | ts=%s | datasets=%s",
            ms.timestamp.isoformat(),
            list(bundle.extras.values.keys()) if bundle.extras else [],
        )


@runs_app.command("list")
def runs_list(
    profile: str = typer.Option("sim", "--profile", "-p", help="Name of config profile."),
    profile_path: str = typer.Option(None, "--profile-path", help="Explicit path to a profile YAML."),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level."),
) -> None:
    setup_logging(level=log_level)
    profile_name, resolved_profile = _resolve_profile_selection(profile, profile_path)
    global_config = load_global_config_from_profile(str(resolved_profile))
    run_cfg = global_config.runlogger
    base_dir = Path(run_cfg.base_dir)
    if not base_dir.is_absolute():
        base_dir = ROOT_DIR / base_dir
    if not base_dir.exists():
        typer.echo(f"No runs directory found at {base_dir}")
        return
    pattern_metrics = run_cfg.filename_patterns.get("metrics", "metrics.json")
    for run_dir in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        metrics_path = run_dir / pattern_metrics
        metrics_info = ""
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
                pf = metrics.get("profit_factor")
                trades = metrics.get("num_trades")
                max_dd = metrics.get("max_drawdown_pct")
                metrics_info = f"PF={pf} trades={trades} maxDD={max_dd}"
            except Exception:
                metrics_info = "metrics:unreadable"
        typer.echo(f"{run_dir.name} | {metrics_info}")


@runs_app.command("metrics")
def runs_metrics(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run identifier (folder name)."),
    profile: str = typer.Option("sim", "--profile", "-p", help="Name of config profile."),
    profile_path: str = typer.Option(None, "--profile-path", help="Explicit path to a profile YAML."),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level."),
) -> None:
    setup_logging(level=log_level)
    _, resolved_profile = _resolve_profile_selection(profile, profile_path)
    global_config = load_global_config_from_profile(str(resolved_profile))
    run_cfg = global_config.runlogger
    base_dir = Path(run_cfg.base_dir)
    if not base_dir.is_absolute():
        base_dir = ROOT_DIR / base_dir
    run_dir = base_dir / run_id
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        raise typer.Exit(code=1)
    metrics_path = run_dir / run_cfg.filename_patterns.get("metrics", "metrics.json")
    if not metrics_path.exists():
        logger.error("metrics.json not found in %s", metrics_path)
        raise typer.Exit(code=1)
    metrics = json.loads(metrics_path.read_text())
    lines = [
        f"Run: {run_id}",
        f"Profit Factor: {metrics.get('profit_factor')}",
        f"Winrate: {metrics.get('winrate')}",
        f"Avg Win: {metrics.get('avg_win')}",
        f"Avg Loss: {metrics.get('avg_loss')}",
        f"Expectancy: {metrics.get('expectancy_per_trade')}",
        f"Trades: {metrics.get('num_trades')}",
        f"Max DD (abs): {metrics.get('max_drawdown_abs')}",
        f"Max DD (%): {metrics.get('max_drawdown_pct')}",
        f"Sharpe-like: {metrics.get('sharpe_like_basic')}",
    ]
    typer.echo("\n".join(lines))


app.add_typer(config_app, name="config")
app.add_typer(extras_app, name="extras")
app.add_typer(runs_app, name="runs")


if __name__ == "__main__":
    app()
