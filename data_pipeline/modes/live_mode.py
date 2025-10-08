from sources.mt5_loader import fetch_mt5
from sources.ccxt_loader import fetch_ccxt
from utils.session_flags import add_session_flags
from utils.validator import validate
from utils.features import add_features
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

def run_live(cfg, exchange=None):
    poll_interval = cfg.get("live_interval_sec", 60)

    while True:
        now = datetime.utcnow()
        start = (now - timedelta(minutes=60)).strftime("%Y-%m-%d %H:%M:%S")
        end = now.strftime("%Y-%m-%d %H:%M:%S")

        for asset in cfg["assets"]:
            symbol = asset["symbol"]
            source = asset["source"]
            timeframes = asset["timeframes"]

            for tf in timeframes:
                print(f"[Live] {symbol} {tf} from {source}...")
                if source == "mt5":
                    df = fetch_mt5(symbol, tf, start, end)
                elif source == "ccxt":
                    df = fetch_ccxt(exchange, symbol, tf, start.split()[0], end.split()[0])
                else:
                    raise ValueError(f"Unknown source {source}")

                df = add_session_flags(df)
                df = validate(df)
                # instrument_cfg aus cfg.assets[x].cost_model ziehen:
                instrument_cfg = asset.get("cost_model", {"type": "forex", "commission": {"kind":"usd_per_lot_roundtrip","value":3.0}})

                df = add_features(df, instrument_cfg=instrument_cfg)

                out_path = Path(cfg["output_dir"]) / f"{symbol.replace('/','')}_{tf}_LIVE.parquet"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if out_path.exists():
                    existing = pd.read_parquet(out_path)
                    df = pd.concat([existing, df]).drop_duplicates("timestamp").sort_values("timestamp")

                df.to_parquet(out_path, index=False)
                print(f"Updated live file {out_path} ({len(df)} rows)")
              
                # zusätzlich zu raw speichern:
                out_feat = Path(cfg["features_dir"]) / f"{symbol.replace('/','')}_{tf}_{cfg['mode'].upper()}.parquet"
                out_feat.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out_feat, index=False)
                print(f"Saved features {len(df)} rows to {out_feat}")

        time.sleep(poll_interval)
