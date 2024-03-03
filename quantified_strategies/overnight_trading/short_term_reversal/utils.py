import datetime as dt
import pandas as pd

try:
    from . import activity as act
except ImportError:
    import activity as act

from quantified_strategies import strategy_utils as utils

ENTER_AT = "Close"
EXIT_AT = "Close"


def fetch(ticker: str) -> pd.DataFrame:
    data = utils.get_data(ticker=ticker)
    data = convert(data=data)
    return data


def convert(data: pd.DataFrame) -> pd.DataFrame:
    
    # Get columns in data i.e. Close
    df = data[["Close", ENTER_AT, EXIT_AT]].copy()
    df.columns = ["Close", "enter", "exit"]

    # Shift exit at backwards
    df["exit_shifted"] = df["exit"].shift(-1)
    
    # Calculate Return
    df["return"] = df["exit_shifted"] / df["enter"] - 1

    return df


def run(ticker: str = None, data: pd.DataFrame = None, full: bool = False, start: dt.date = None, end: dt.date = None) -> pd.DataFrame:

    if data is None:
        assert ticker is not None
        data = fetch(ticker=ticker)
        
        if start is not None:
            data = data.loc[data.index.date >= start]
        if end is not None:
            data = data.loc[data.index.date <= end]
    else:
        data = convert(data=data)
    
    data["active"] = act.get_activity(data=data)
    data["strat_ret"] = data["active"] * data["return"]

    if not full:
        return data["active"].replace(False, None) * data["strat_ret"]
    
    data["cum_strat_ret"] = utils.get_cumulative_return(returns=data["strat_ret"], total=False)
    data["cum_hodl_ret"] = utils.get_cumulative_return(returns=data["return"], total=False)
    data["enter_flag"] = data["active"].astype(int).diff().clip(lower=0.0).fillna(0.0).astype(bool)
    data["trade_number"] = data["enter_flag"].cumsum()
        
    return data
