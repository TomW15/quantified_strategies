import datetime as dt
import pandas as pd

try:
    from . import available as avb
except ImportError:
    import available as avb
    

def is_active(data: pd.DataFrame = None, date: dt.date = None, close_on_nth_day_in_new_year: int = avb.THIRD_TRADING_DAY) -> bool:
    return avb.is_available(date=date, close_on_nth_day_in_new_year=close_on_nth_day_in_new_year)


def get_activity(data: pd.DataFrame, close_on_nth_day_in_new_year: int = avb.THIRD_TRADING_DAY) -> pd.Series:
    return avb.get_availability(start=data.index[0], end=data.index[-1], close_on_nth_day_in_new_year=close_on_nth_day_in_new_year)
