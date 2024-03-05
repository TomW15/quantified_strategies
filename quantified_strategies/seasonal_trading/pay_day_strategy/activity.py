import datetime as dt
import pandas as pd

try:
    from . import available as avb
except ImportError:
    import available as avb
    

def is_active(data: pd.DataFrame = None, date: dt.date = None, buy_nth_last_day: int = avb.BUY_NTH_LAST_DAY, 
              sell_nth_first_day: int = avb.SELL_NTH_FIRST_DAY) -> bool:
    return avb.is_available(date=date, buy_nth_last_day=buy_nth_last_day, sell_nth_first_day=sell_nth_first_day)


def get_activity(data: pd.DataFrame, buy_nth_last_day: int = avb.BUY_NTH_LAST_DAY, sell_nth_first_day: int = avb.SELL_NTH_FIRST_DAY) -> pd.Series:
    return avb.get_availability(start=data.index[0], end=data.index[-1], buy_nth_last_day=buy_nth_last_day, sell_nth_first_day=sell_nth_first_day)
