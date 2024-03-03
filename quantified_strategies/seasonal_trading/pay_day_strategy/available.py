import datetime as dt
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)


def is_available(date: dt.date = None) -> bool:

    if date is None:
        date = dt.date.today()
    
    date = pd.Timestamp(date)
    
    bus_days = pd.date_range(start=date-dt.timedelta(days=10), end=date+dt.timedelta(days=10), freq="B")

    year = date.year
    month = date.month

    # Is first trading day of month
    first_date_of_month = dt.date(year, month, 1)
    first_n_business_days_of_month = pd.date_range(start=first_date_of_month, end=first_date_of_month+dt.timedelta(days=7), freq="B")
    if first_n_business_days_of_month[0] == date:
        return True
    
    # Is in last 5 trading days of month
    first_date_of_next_month = dt.date((year + 1) if month == 12 else year, month % 12 + 1, 1)
    last_date_of_month = first_date_of_next_month - dt.timedelta(days=1)
    last_n_business_days_of_month = pd.date_range(start=last_date_of_month-dt.timedelta(days=15), end=last_date_of_month, freq="B")
    last_5_business_days_of_month = last_n_business_days_of_month[-5:]
    if date in last_5_business_days_of_month:
        return True

    # If date is a weekend at end of month i.e. after first of the last 5 business days
    if date > last_5_business_days_of_month[0]:
        return True
    # If date is a weekend at start of month
    if date < first_n_business_days_of_month[0]:
        return True
    
    return False


def get_availability(start: dt.date = dt.date(2000, 1, 1), end: dt.date = dt.date.today()) -> pd.Series:
    dates = pd.date_range(start=start, end=end)
    bus_days = pd.date_range(start=start, end=end, freq="B")
    availability = pd.Series({date: is_available(date=date) for date in bus_days}, dtype=bool)
    return availability.reindex(index=dates).ffill().dropna()
