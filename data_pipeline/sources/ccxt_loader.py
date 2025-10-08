import ccxt
import pandas as pd

def init_ccxt(exchange_id: str, api_key: str, secret: str):
    ex_class = getattr(ccxt, exchange_id)
    return ex_class({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True
    })

def fetch_ccxt(exchange, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    since = exchange.parse8601(start + "T00:00:00Z")
    all_data = []
    limit = 1000
    while since:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if since > exchange.parse8601(end + "T00:00:00Z"):
            break

    df = pd.DataFrame(all_data, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df
