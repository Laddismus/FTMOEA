from sources.mt5_loader import fetch_mt5
from sources.ccxt_loader import fetch_ccxt
from utils.session_flags import add_session_flags
from utils.validator import validate
from utils.features import add_features
import pandas as pd
from pathlib import Path

def run_backtest(cfg, exchange=None):
    for asset in cfg["assets"]:
        symbol = asset["symbol"]
        source = asset["source"]
        timeframes = asset["timeframes"]

        for tf in timeframes:
            print(f"[Backtest] {symbol} {tf} from {source}...")
            if source == "mt5":
                df = fetch_mt5(symbol, tf, cfg["start"], cfg["end"])
            elif source == "ccxt":
                df = fetch_ccxt(exchange, symbol, tf, cfg["start"], cfg["end"])
            else:
                raise ValueError(f"Unknown source {source}")

            df = add_session_flags(df)
            df = validate(df)
            

            # instrument_cfg aus cfg.assets[x].cost_model ziehen:
            instrument_cfg = asset.get("cost_model", {"type": "forex", "commission": {"kind":"usd_per_lot_roundtrip","value":3.0}})

            df = add_features(df, instrument_cfg=instrument_cfg)
            out_path = Path(cfg["output_dir"]) / f"{symbol.replace('/','')}_{tf}_{cfg['start']}_{cfg['end']}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, index=False)
            print(f"Saved {len(df)} rows to {out_path}")

            # zusätzlich zu raw speichern:
            out_feat = Path(cfg["features_dir"]) / f"{symbol.replace('/','')}_{tf}_{cfg['mode'].upper()}.parquet"
            out_feat.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_feat, index=False)
            print(f"Saved features {len(df)} rows to {out_feat}")
