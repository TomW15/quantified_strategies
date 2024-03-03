import datetime as dt
import pandas as pd

try:
    from . import available as avb
except ImportError:
    import available as avb
    

def is_active(data: pd.DataFrame = None, date: dt.date = None) -> bool:
    return avb.is_available(date=date)


def get_activity(data: pd.DataFrame) -> pd.Series:
    return avb.get_availability(start=data.index[0], end=data.index[-1])
