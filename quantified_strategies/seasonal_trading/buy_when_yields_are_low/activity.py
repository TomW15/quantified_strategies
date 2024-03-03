import datetime as dt
import numpy as np
import pandas as pd

try:
    from . import utils
except ImportError:
    import utils

WINDOW: int = 15


def is_active(treasury: pd.DataFrame, data: pd.DataFrame = None, date: dt.date = None, window: int = WINDOW) -> bool:

    date = dt.date.today() if date is None else date
    
    sample_treasury = treasury.loc[treasury.index.date <= date]
    activity = get_activity(data=None, treasury=sample_treasury, window=window)
    
    return activity.iloc[-1]


def get_activity(treasury: pd.DataFrame, data: pd.DataFrame = None, window: int = WINDOW) -> pd.Series:

    ## Rules
    
    # - Buy S&P 500 at the close when the 10 year yield drops below its 15-day exponential moving average; and
    # - Exit at the close when the yield crosses above its 15-day exponential moving average.
        
    return utils.get_position(treasury=treasury, window=WINDOW, reverse=False)
