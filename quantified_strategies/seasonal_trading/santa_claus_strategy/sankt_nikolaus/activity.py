import datetime as dt
import pandas as pd

try:
    from . import available as avb
except ImportError:
    import available as avb
    

def is_active(data: pd.DataFrame = None, date: dt.date = None, entry_day: int = avb.FRIDAY, nth_entry_day: int = avb.THIRD) -> bool:
    return avb.is_available(date=date, entry_day=entry_day, nth_entry_day=nth_entry_day)


def get_activity(data: pd.DataFrame, entry_day: int = avb.FRIDAY, nth_entry_day: int = avb.THIRD) -> pd.Series:
    return avb.get_availability(start=data.index[0], end=data.index[-1], entry_day=entry_day, nth_entry_day=nth_entry_day)
