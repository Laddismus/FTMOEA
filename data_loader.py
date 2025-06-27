import pandas as pd

def load_local_price_data(filename):
    df = pd.read_csv(filename, sep="\t")
    df.columns = [col.replace("<", "").replace(">", "").lower() for col in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    df = df.rename(columns={'close': 'close', 'open': 'open', 'high': 'high', 'low': 'low', 'tickvol': 'volume'})
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df
