import datetime as dt
import numpy as np
import typing as t


# Default Interval to use on init
DEFAULT_INTERVAL: str = "1h"
MIN_INTERVAL: str = "1m"

# Available Intervals in yahoo finance
INTERVALS: list[str] = [
    "3mo",
    "1mo",
    "1wk",
    "5d",
    "1d",
    "1h",
    "90m",
    "30m",
    "15m",
    "5m",
    "2m",
    "1m",
]

HISTORY: dict[str, int] = {
    "1h": 730,
    "90m": 60,
    "30m": 60,
    "15m": 60,
    "5m": 60,
    "2m": 60,
    "1m": 30,
}
HISTORY_MAX: int = (dt.date.today() - dt.date(2000, 1, 1)).days

TOTAL: dict[str, int] = {
    "1h": 730,
    "90m": 60,
    "30m": 60,
    "15m": 60,
    "5m": 60,
    "2m": 60,
    "1m": 7,
}
TOTAL_MAX: int = (dt.date.today() - dt.date(2000, 1, 1)).days


def fetch_intervals() -> t.List:
    
    intervals_ = []
    MAX_DATE = dt.date.today()

    for interval in INTERVALS[::-1]:
        max_history = HISTORY.get(interval, HISTORY_MAX) - 2
        max_request = TOTAL.get(interval, TOTAL_MAX) - 2
        N = int(np.ceil(max_history / max_request))

        start_date = dt.date.today() - dt.timedelta(days=max_history)

        for n in range(N):
            if MAX_DATE == start_date:
                break

            if start_date + dt.timedelta(days=(n + 1) * max_request) > MAX_DATE:
                intervals_.append(
                    (
                        interval,
                        start_date + dt.timedelta(days=n * max_request),
                        MAX_DATE,
                    )
                )
                MAX_DATE = start_date

            else:
                intervals_.append(
                    (
                        interval,
                        start_date + dt.timedelta(days=n * max_request),
                        start_date + dt.timedelta(days=(n + 1) * max_request),
                    )
                )

        if MAX_DATE == dt.date(2000, 1, 1):
            break

    return intervals_


def fetch_latest(interval: str) -> dt.date:
    max_request = TOTAL.get(interval, TOTAL_MAX) - 1
    return dt.date.today() - dt.timedelta(days=max_request)


if __name__ == "__main__":
    intervals = fetch()
    latest = fetch_latest(interval="1m")
