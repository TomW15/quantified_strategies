import datetime as dt
import pandas as pd
import typing as t

try:
    from . import available as avb
except ImportError:
    import available as avb
    

def is_active(data: pd.DataFrame = None, date: dt.date = None, enter_before: t.Dict[str, int] = avb.MARCH_31ST, exit_after: t.Dict[str, int] = avb.APRIL_15TH) -> bool:
    return avb.is_available(date=date, enter_before=enter_before, exit_after=exit_after)


def get_activity(data: pd.DataFrame, enter_before: t.Dict[str, int] = avb.MARCH_31ST, exit_after: t.Dict[str, int] = avb.APRIL_15TH) -> pd.Series:
    return avb.get_availability(start=data.index[0], end=data.index[-1], enter_before=enter_before, exit_after=exit_after)
