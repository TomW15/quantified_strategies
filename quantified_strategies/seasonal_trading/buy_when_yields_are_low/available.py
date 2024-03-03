import datetime as dt
import pandas as pd


def is_available(date: dt.date = None) -> bool:
    return True


def get_availability(start: dt.date = dt.date(2000, 1, 1), end: dt.date = dt.date.today()) -> pd.Series:
    return pd.Series(True, index=pd.date_range(start=start, end=end))
