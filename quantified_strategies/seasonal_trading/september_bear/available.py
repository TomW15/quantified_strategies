import datetime as dt
import pandas as pd

THIRD_WEEK: int = 3
COVER_DURATION: int = 1
FRIDAY: int = 4


def is_available(date: dt.date | pd.Timestamp = None, xday: int = FRIDAY, week_enter: int = THIRD_WEEK, cover_duration: int = COVER_DURATION) -> bool:

    if date is None:
        date = dt.date.today()

    if isinstance(date, pd.Timestamp):
        date = date.date()

    if date.month != 9:
        return False
    
    sept1 = dt.date(date.year, 9, 1)
    monday_before_sept1 = sept1 - dt.timedelta(days=sept1.weekday())
    xday_after_monday_before_sept1 = monday_before_sept1 + dt.timedelta(days=xday)
    xday_after_sept1 = (xday_after_monday_before_sept1 + dt.timedelta(days=7)) if xday_after_monday_before_sept1 < sept1 else xday_after_monday_before_sept1

    nth_xday_of_sept = xday_after_sept1 + dt.timedelta(days=7 * (week_enter - 1))
    nth_xday_of_sept_plus_1_week = nth_xday_of_sept + dt.timedelta(days=7 * cover_duration)
    
    if date < nth_xday_of_sept:
        return False
    if date >= nth_xday_of_sept_plus_1_week:
        return False
    return True


def get_availability(start: dt.date = dt.date(2000, 1, 1), end: dt.date = dt.date.today(),
                    xday: int = FRIDAY, week_enter: int = THIRD_WEEK, cover_duration: int = COVER_DURATION) -> pd.Series:
    return pd.Series({date: is_available(date=date, xday=xday, week_enter=week_enter, cover_duration=cover_duration) for date in pd.date_range(start=start, end=end)})
    