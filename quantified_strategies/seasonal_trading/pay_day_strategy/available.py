import datetime as dt
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

BUY_NTH_LAST_DAY: int = 5
SELL_NTH_FIRST_DAY: int = 3


def is_available(date: dt.date, buy_nth_last_day: int = BUY_NTH_LAST_DAY, sell_nth_first_day: int = SELL_NTH_FIRST_DAY) -> bool:

    assert buy_nth_last_day > 0
    assert sell_nth_first_day > 0
    
    if date is None:
        date = dt.date.today()
    
    date = pd.Timestamp(date)

    year = date.year
    month = date.month

    # Is first trading day of month
    first_date_of_month = dt.date(year, month, 1)
    first_n_business_days_of_month = pd.date_range(start=first_date_of_month, end=first_date_of_month+dt.timedelta(days=sell_nth_first_day*3), freq="B")
    if date <= first_n_business_days_of_month[sell_nth_first_day-1]:
        return True
    
    # Is in last 5 trading days of month
    first_date_of_next_month = dt.date((year + 1) if month == 12 else year, month % 12 + 1, 1)
    last_date_of_month = first_date_of_next_month - dt.timedelta(days=1)
    last_n_business_days_of_month = pd.date_range(start=last_date_of_month-dt.timedelta(days=buy_nth_last_day*3), end=last_date_of_month, freq="B")
    last_n_business_days_of_month = last_n_business_days_of_month[-buy_nth_last_day:]
    if date >= last_n_business_days_of_month[0]:
        return True
    
    return False


def get_availability(start: dt.date = dt.date(2000, 1, 1), end: dt.date = dt.date.today(), buy_nth_last_day: int = BUY_NTH_LAST_DAY, 
                     sell_nth_first_day: int = SELL_NTH_FIRST_DAY) -> pd.Series:
    dates = pd.date_range(start=start, end=end)
    bus_days = pd.date_range(start=start, end=end, freq="B")
    availability = pd.Series({date: is_available(date=date, buy_nth_last_day=buy_nth_last_day, sell_nth_first_day=sell_nth_first_day) for date in bus_days}, dtype=bool)
    return availability.reindex(index=dates).ffill().dropna()
