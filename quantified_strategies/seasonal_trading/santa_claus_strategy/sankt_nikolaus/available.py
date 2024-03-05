import datetime as dt
import pandas as pd
import pandas_market_calendars as mcal

# Create a calendar
EUREX = mcal.get_calendar("EUREX")
FRIDAY: int = 4
THIRD: int = 3


def is_available(date: dt.date | pd.Timestamp = None, entry_day: int = FRIDAY, nth_entry_day: int = THIRD) -> bool:

    if date is None:
        date = dt.date.today()

    if isinstance(date, pd.Timestamp):
        date = date.date()
    
    if date.month != 12:
        return False

    # Month is December
    # Define dec 1 of year
    dec1 = dt.date(date.year, 12, 1)

    ### Fetch nth Entry Day in December
    # Fetch start of week containing dec1
    monday_before_dec1 = dec1 - dt.timedelta(days=dec1.weekday())
    # Fetch Entry Day after start of week containing dec1
    entry_day_after_monday_before_dec1 = monday_before_dec1 + dt.timedelta(days=entry_day)
    # If end of business week containing dec14 is before dec14 i.e. dec14 is Sat/Sun, then add a week
    if entry_day_after_monday_before_dec1 < dec1:
        entry_day_after_dec1 = entry_day_after_monday_before_dec1 + dt.timedelta(days=7)
    else:
        entry_day_after_dec1 = entry_day_after_monday_before_dec1
        
    # Define third Friday in December
    nth_entry_day_of_dec = entry_day_after_dec1 + dt.timedelta(days=7 * (nth_entry_day - 1))

    ### Fetch last trading day of year

    # Define Xmas and Hogmonay of date's year
    dec25 = dt.date(date.year, 12, 25)
    dec31 = dt.date(date.year, 12, 31)
    # Fetch trading hours in week folling Xmas
    trading_hours = EUREX.schedule(start_date=dec25, end_date=dec31)
    # Trading days following Xmas
    trading_days_after_Xmas = [date.date() for date in mcal.date_range(trading_hours, frequency='1D')]
    # Last trading day before New Year
    last_trading_day_before_new_year = trading_days_after_Xmas[-1]

    # If date in interval after third Friday of December and before last trading day of year
    return (date >= nth_entry_day_of_dec) & (date < last_trading_day_before_new_year)


def get_availability(start: dt.date = dt.date(2000, 1, 1), end: dt.date = dt.date.today(), entry_day: int = FRIDAY, nth_entry_day: int = THIRD) -> pd.Series:
    dates = pd.date_range(start=start, end=end)
    return pd.Series({date: is_available(date=date, entry_day=entry_day, nth_entry_day=nth_entry_day) for date in dates}, dtype=bool)
