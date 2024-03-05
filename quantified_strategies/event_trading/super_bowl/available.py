import datetime as dt
from loguru import logger
import pandas as pd
import typing as t

try:
    from . import utils
except ImportError:
    import utils

HOLD_PERIOD = 15


def _get_hold_period(super_bowl_date: dt.date, hold_period: int = HOLD_PERIOD) -> t.List[dt.date]:
    return [date.date() for date in pd.date_range(start=super_bowl_date, end=super_bowl_date+dt.timedelta(days=hold_period))]


def is_available(date: dt.date = None, hold_period: int = HOLD_PERIOD) -> bool:

    if date is None:
        date = dt.date.today()
    
    # Get Published Super Bowl Dates
    SUPER_BOWL_DATES = utils.get_super_bowl_dates()
    # Map Super Bowls to a year
    SUPER_BOWLS_BY_YEAR = {date.year: date for date in SUPER_BOWL_DATES}
    # Fetch current year's Super Bowl date
    date_year_super_bowl_date = SUPER_BOWLS_BY_YEAR.get(date.year)

    if date_year_super_bowl_date is None:
        logger.info("Current year's Super Bowl has not been announced.")
        return False

    # Get a list of days to position is available
    hold_period = _get_hold_period(super_bowl_date=date_year_super_bowl_date, hold_period=hold_period)

    # Check if current date is in hold period
    is_available_ = date in hold_period

    return is_available_


def get_availability(start: dt.date = dt.date(1950, 1, 1), end: dt.date = dt.date.today()) -> pd.Series:

    assert start <= end, f"'start' must be before 'end'"
    
    # Get years in between start and end
    years = [date for date in range(start.year, end.year + 1)]
    
    # Get Published Super Bowl Dates
    SUPER_BOWL_DATES = utils.get_super_bowl_dates()
    
    # Map Super Bowls to a year
    SUPER_BOWLS_BY_YEAR = {date.year: date for date in SUPER_BOWL_DATES if date.year in years}
    
    if len(SUPER_BOWLS_BY_YEAR) == 0:
        logger.info(f"No Super Bowls announced for dates {start}->{end}.")
        return pd.Series(False, index=pd.date_range(start=start, end=end), dtype=bool)
    
    hold_periods = sum([_get_hold_period(super_bowl_date=sb_date, hold_period=HOLD_PERIOD) for _, sb_date in SUPER_BOWLS_BY_YEAR.items()], [])
    hold_periods = pd.DatetimeIndex(hold_periods)
    
    availability = pd.Series(True, index=hold_periods, dtype=bool)
    availability = availability.reindex(index=pd.date_range(start=start, end=end).union(hold_periods), fill_value=False)

    return availability

