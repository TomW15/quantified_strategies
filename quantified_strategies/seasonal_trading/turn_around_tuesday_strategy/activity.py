import datetime as dt
import pandas as pd

try:
    from . import utils
except ImportError:
    import utils


def get_entry_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["is_monday"] = df.index.weekday == 0
    df["price_less_than_yday_price"] = df["asset"] < df["asset"].shift(1)
    df["enter_flag"] = df["is_monday"] & df["price_less_than_yday_price"]
    return df

def get_exit_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["price_today_higher_than_yday_high"] = df["asset"] > df["asset_high"].shift(1)
    df["five_trading_days_later"] = df["enter_flag"].shift(5)
    df["exit_flag"] = df["price_today_higher_than_yday_high"] | df["five_trading_days_later"]
    return df

def calculate_positions(df: pd.DataFrame) -> pd.DataFrame:
    
    def get_position(row: pd.Series, last_position: int) -> int:
        if row["enter_flag"]:
            return 1
        if last_position == 0:
            return 0
        if row["exit_flag"]:
            return 0
        return last_position
    
    positions = dict()
    pos = 0
    for date, row in df.iterrows():
        pos = get_position(row=row, last_position=pos)
        positions[date] = pos

    df["pos"] = pd.Series(positions, dtype=int)
    return df


def is_active(data: pd.DataFrame, date: dt.date = None) -> bool:

    date = dt.date.today() if date is None else date
    
    sample_data = data.loc[data.index.date <= date].copy()
    sample_data = sample_data.tail(7)
    
    activity = get_activity(data=sample_data)
    
    return activity.iloc[-1]


def get_activity(data: pd.DataFrame) -> pd.Series:

    ## Rules
    
    # 1. Today must be a Monday
    # 2. The Close must be lower than Friday's close
    # 3. If 1 and 2 are true, then buy at the close
    # 4. Sell when today's close is higher than yesterday's high or after 5 trading days.

    # Get Entry Flags
    data = get_entry_flags(df=data)
    # Get Exit Flags
    data = get_exit_flags(df=data)
    # Calculate Positions
    data = calculate_positions(df=data)

    active = data["pos"].astype(bool)
    
    return active
