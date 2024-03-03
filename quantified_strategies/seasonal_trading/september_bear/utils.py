import datetime as dt
import pandas as pd

try:
    from . import activity as act
except ImportError:
    import activity as act

from quantified_strategies import strategy_utils as utils


def run(ticker: str = None, data: pd.DataFrame = None, full: bool = False, start: dt.date = None, end: dt.date = None, **kwargs) -> pd.DataFrame:

    if data is None:
        assert ticker is not None
        data = utils.get_data(ticker=ticker, columns="Adj Close")
        if start is not None:
            data = data.loc[data.index.date >= start]
        if end is not None:
            data = data.loc[data.index.date <= end]
        data = data.to_frame(name="asset")
    
    data["ret"] = data["asset"].pct_change()
    data["ret_shifted"] = data["ret"].shift(-1)
    data["active"] = act.get_activity(data=data, **kwargs)
    data["strat_ret"] = data["active"] * -data["ret_shifted"]

    if not full:
        return data["active"].replace(False, None) * data["strat_ret"]
    
    data["cum_strat_ret"] = utils.get_cumulative_return(returns=data["strat_ret"], total=False)
    data["cum_hodl_ret"] = utils.get_cumulative_return(returns=data["ret_shifted"], total=False)
    data["enter_flag"] = data["active"].astype(int).diff().clip(lower=0.0).fillna(0.0).astype(bool)
    data["trade_number"] = data["enter_flag"].cumsum()
        
    return data
