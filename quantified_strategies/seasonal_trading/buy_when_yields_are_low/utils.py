import datetime as dt
import numpy as np
import pandas as pd

try:
    from . import activity as act
except ImportError:
    import activity as act

from quantified_strategies import strategy_utils as utils

EMA_WINDOW_SIZE = 15
MAJORITY_IN_N = 5
MAJORITY = np.ceil(MAJORITY_IN_N / 2) / MAJORITY_IN_N


def fetch(ticker: str) -> pd.DataFrame:
    data = utils.get_data(ticker=ticker, columns="Close", flt=False)
    data = data.to_frame(name="asset")
    return data


def fetch_treasury() -> pd.DataFrame:
    return utils.get_data(ticker="^TNX", columns="Close", source="yahoo")


def calculate_ema(data: pd.Series, window: int = EMA_WINDOW_SIZE) -> pd.Series:
    # Calculate the exponential moving average
    ema = data.ewm(span=window, adjust=False).mean()
    return ema


def calc_signal(x: pd.Series) -> int:
    if np.mean(x) > MAJORITY:
        return 1
    if np.mean(x) < (1 - MAJORITY):
        return 0
    return np.nan


def get_position(treasury: pd.Series, window: int = EMA_WINDOW_SIZE, reverse: bool = False) -> pd.Series:
    
    treasury_moving_average = calculate_ema(data=treasury, window=window)

    if reverse:
        signal = (treasury > treasury_moving_average).astype(int)
    else:
        signal = (treasury < treasury_moving_average).astype(int)
    
    signal = signal.rolling(window=MAJORITY_IN_N).apply(lambda x: calc_signal(x=x)).ffill().fillna(0)
    signal = signal.astype(bool)
    
    return signal


def run(ticker: str = None, data: pd.DataFrame = None, full: bool = False, start: dt.date = None, end: dt.date = None, asset_day_after: bool = False) -> pd.DataFrame:

    treasury = fetch_treasury()
    
    if data is None:
        assert ticker is not None
        data = fetch(ticker=ticker)

    if start is not None:
        data = data.loc[data.index.date >= start]
        treasury = treasury.loc[treasury.index.date >= start]
    if end is not None:
        data = data.loc[data.index.date <= end]
        treasury = treasury.loc[treasury.index.date <= end]

    data = data.reindex(index=data.index.union(treasury.index), method="ffill")
    treasury = treasury.reindex(index=data.index.union(treasury.index), method="ffill")
    
    data["active"] = act.get_activity(data=data, treasury=treasury)
    data["ret"] = data["asset"].pct_change()
    data["ret_shifted"] = data["ret"].shift(-1)
    if asset_day_after:
        data["ret_shifted"] = data["ret_shifted"].shift(-1)
    data["strat_ret"] = data["active"] * data["ret_shifted"]

    if not full:
        return data["active"].replace(False, None) * data["strat_ret"]
    
    data["cum_strat_ret"] = utils.get_cumulative_return(returns=data["strat_ret"], total=False)
    data["cum_hodl_ret"] = utils.get_cumulative_return(returns=data["ret_shifted"], total=False)
    data["enter_flag"] = data["active"].astype(int).diff().clip(lower=0.0).fillna(0.0).astype(bool)
    data["trade_number"] = data["enter_flag"].cumsum()
        
    return data

