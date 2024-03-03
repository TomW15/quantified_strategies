import datetime as dt
import pandas as pd


def is_available(date: dt.date = None) -> bool:
    
    if date is None:
        date = dt.date.today()
    date = pd.Timestamp(date)
    
    return get_availability(start=date, end=date)[date]


def get_availability(start: dt.date = dt.date(2000, 1, 1), end: dt.date = dt.date.today()) -> pd.Series:

    # Get all dates between start and end
    DATES = pd.date_range(start=start, end=end)
    # Define year to determine if first trading day in July as been seen per year
    year = 0
    # Define availability as dictionary: date to boolean
    available = dict()
    # Loop over dates
    for date in DATES:

        # Is date in June and after 23rd
        if (date.month == 6) and (date.day >= 23):
            available[date] = True
        # Is date in July and first trading day of July unseen
        elif (date.month == 7) and (year != date.year):
            available[date] = True
            # Is date a trading day
            if date.weekday() not in [5, 6]:
                # Update year to date's year
                year = date.year
        else:
            available[date] = False

    availability = pd.Series(available, dtype=bool)
    
    return availability
