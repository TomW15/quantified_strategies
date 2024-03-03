import datetime as dt
import pandas as pd

try:
    from . import available as avb
except ImportError:
    import available as avb
    

def is_active(data: pd.DataFrame = None, date: dt.date = None, xday: int = avb.FRIDAY, week_enter: int = avb.THIRD_WEEK, cover_duration: int = avb.COVER_DURATION) -> bool:
    return avb.is_available(date=date, xday=xday, week_enter=week_enter, cover_duration=cover_duration)


def get_activity(data: pd.DataFrame, xday: int = avb.FRIDAY, week_enter: int = avb.THIRD_WEEK, cover_duration: int = avb.COVER_DURATION) -> pd.Series:
    return avb.get_availability(start=data.index[0], end=data.index[-1], xday=xday, week_enter=week_enter, cover_duration=cover_duration)
