import yaml
from sources.mt5_loader import init_mt5, shutdown_mt5
from sources.ccxt_loader import init_ccxt
from modes.backtest_mode import run_backtest
from modes.live_mode import run_live

def run_pipeline(cfg_path="data_config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))

    # Init sources
    if "mt5" in cfg:
        init_mt5(**cfg["mt5"])
    exchange = None
    if "ccxt" in cfg:
        exchange = init_ccxt(**cfg["ccxt"])

    if cfg["mode"] == "backtest":
        run_backtest(cfg, exchange)
    elif cfg["mode"] == "live":
        run_live(cfg, exchange)
    else:
        raise ValueError(f"Unknown mode {cfg['mode']}")

    if "mt5" in cfg:
        shutdown_mt5()

if __name__ == "__main__":
    run_pipeline()
