import datetime as dt
import pandas as pd

try:
    from . import utils
except ImportError:
    import utils
    
N_BEAR_DAYS = 3


def is_active(data: pd.DataFrame, date: dt.date = None) -> bool:

    date = dt.date.today() if date is None else date
    
    sample_data = data.loc[data.index.date <= date].copy()
    sample_data = sample_data.tail(N_BEAR_DAYS+1)
    
    activity = get_activity(data=sample_data)
    
    return activity.iloc[-1]


def get_activity(data: pd.DataFrame) -> pd.Series:

    ## Rules
    
    # 1. SPY must be down three days in a row (close to close).
    # 2. Entry on the third down day at the close.
    # 3. Exit on the open the next day
    
    # Check if asset is down verus yesterday
    data["down_from_yday"] = data["Close"] < data["Close"].shift(1)
    # Calculate number of down days in last 'N_BEAR_DAYS' days
    data["down_xdays_in_a_row"] = data["down_from_yday"].rolling(window=N_BEAR_DAYS).sum().fillna(0.0)
    # Check if asset has had 'N_BEAR_DAYS' in a row
    active = data["down_xdays_in_a_row"] >= N_BEAR_DAYS

    return active
