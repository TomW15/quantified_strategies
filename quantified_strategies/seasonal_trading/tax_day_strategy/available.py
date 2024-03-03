import datetime as dt
import pandas as pd
import pandas_market_calendars as mcal
import typing as t

# Create a calendar
NYSE = mcal.get_calendar('NYSE')
MARCH_31ST = dict(month=3, day=31)
APRIL_15TH = dict(month=4, day=15)

print("May need to change to fit other strategy")


def is_available(date: dt.date | pd.Timestamp = None, enter_before: t.Dict[str, int] = MARCH_31ST, exit_after: t.Dict[str, int] = APRIL_15TH) -> bool:

    def get_last_trading_day_after():

        earliest_enter_day = dt.date(date.year, **enter_before)
        earliest_enter_day_minus_1w = earliest_enter_day - dt.timedelta(days=7)
        
        NYSE_TRADING_HOURS = NYSE.schedule(start_date=earliest_enter_day_minus_1w, end_date=earliest_enter_day)
        NYSE_TRADING_DAYS = mcal.date_range(NYSE_TRADING_HOURS, frequency='1D')
        last_trading_day_after = NYSE_TRADING_DAYS[-1].date()
        
        return last_trading_day_after

    def get_first_trading_day_after():
        
        earliest_exit_day = dt.date(date.year, **exit_after)
        earliest_exit_day_plus_1w = earliest_exit_day + dt.timedelta(days=7)
        
        NYSE_TRADING_HOURS = NYSE.schedule(start_date=earliest_exit_day, end_date=earliest_exit_day_plus_1w)
        NYSE_TRADING_DAYS = mcal.date_range(NYSE_TRADING_HOURS, frequency='1D')
        friday_trading_day_after = NYSE_TRADING_DAYS[0].date()
        
        return friday_trading_day_after

    if date is None:
        date = dt.date.today()
    
    if isinstance(date, pd.Timestamp):
        date = date.date()
        
    if enter_before["month"] <= date.month <= exit_after["month"]:
        pass
    else:
        return False

    enter_day = get_last_trading_day_after()
    exit_day = get_first_trading_day_after()

    if enter_day <= date < exit_day:
        return True
    return False


def get_availability(start: dt.date = dt.date(2000, 1, 1), end: dt.date = dt.date.today(),
                    enter_before: t.Dict[str, int] = MARCH_31ST, exit_after: t.Dict[str, int] = APRIL_15TH) -> pd.Series:
    return pd.Series({date: is_available(date=date, enter_before=enter_before, exit_after=exit_after) for date in pd.date_range(start=start, end=end)})
