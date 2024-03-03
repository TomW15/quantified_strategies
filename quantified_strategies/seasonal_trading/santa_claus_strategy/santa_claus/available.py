import datetime as dt
import pandas as pd
import pandas_market_calendars as mcal

# Create a calendar
NYSE = mcal.get_calendar("NYSE")
THIRD_TRADING_DAY = 3


def is_available(date: dt.date | pd.Timestamp = None, close_on_nth_day_in_new_year: int = THIRD_TRADING_DAY) -> bool:

    if date is None:
        date = dt.date.today()

    if isinstance(date, pd.Timestamp):
        date = date.date()
    
    if date.month not in [12, 1]:
        return False

    # Is date in December
    if date.month == 12:
        # Is date before the 14th
        if date.day < 14:
            return False
        
        # Define dec 14 of year
        dec14 = dt.date(date.year, 12, 14)
        # Is December 14th a Friday?
        if dec14.weekday() == 4:
            return True
            
        # Fetch start of week containing dec14
        monday_before_dec14 = dec14 - dt.timedelta(days=dec14.weekday())
        # Fetch end of business week containing dec14
        friday_after_monday_before_dec14 = monday_before_dec14 + dt.timedelta(days=4)
        # If end of business week containing dec14 is before dec14 i.e. dec14 is Sat/Sun, then add a week
        if friday_after_monday_before_dec14 < dec14:
            friday_after_dec14 = friday_after_monday_before_dec14 + dt.timedelta(days=7)
        else:
            friday_after_dec14 = friday_after_monday_before_dec14
        # If date is after friday after dec14
        return date >= friday_after_dec14

    # date is in January

    # Define first week of New Year
    jan1 = dt.date(date.year, 1, 1)
    jan15 = jan1 + dt.timedelta(days=14)
    # Fetch trading hours in first week of New Year
    trading_hours = NYSE.schedule(start_date=jan1, end_date=jan15)
    # First trading days of New Year
    first_trading_days_of_new_year = [date.date() for date in mcal.date_range(trading_hours, frequency='1D')]
    # First trading day of New Year
    nth_trading_day_of_new_year = first_trading_days_of_new_year[close_on_nth_day_in_new_year-1]
    
    if date < nth_trading_day_of_new_year:
        return True
        
    return False


def get_availability(start: dt.date = dt.date(2000, 1, 1), end: dt.date = dt.date.today(), 
                     close_on_nth_day_in_new_year: int = THIRD_TRADING_DAY) -> pd.Series:
    dates = pd.date_range(start=start, end=end)
    return pd.Series({date: is_available(date=date, close_on_nth_day_in_new_year=close_on_nth_day_in_new_year) for date in dates}, dtype=bool)
