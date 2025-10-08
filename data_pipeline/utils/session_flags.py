import pandas as pd

def add_session_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour
    df["session"] = "Off"
    df.loc[df["hour"].between(0,7), "session"] = "Asia"
    df.loc[df["hour"].between(8,15), "session"] = "London"
    df.loc[df["hour"].between(13,20), "session"] = "NewYork"
    return df.drop(columns="hour")
