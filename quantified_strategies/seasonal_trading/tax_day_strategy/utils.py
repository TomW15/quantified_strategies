# packages

from collections import OrderedDict
import cvxpy as cp
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import typing as t
import warnings
import yfinance as yf

try:
    from . import constants
except ImportError:
    import constants

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
    data["strat_ret"] = data["active"] * data["ret_shifted"]

    if not full:
        return data["active"].replace(False, None) * data["strat_ret"]
    
    data["cum_strat_ret"] = utils.get_cumulative_return(returns=data["strat_ret"], total=False)
    data["cum_hodl_ret"] = utils.get_cumulative_return(returns=data["ret_shifted"], total=False)
    data["enter_flag"] = data["active"].astype(int).diff().clip(lower=0.0).fillna(0.0).astype(bool)
    data["trade_number"] = data["enter_flag"].cumsum()
        
    return data


def get_returns(df: pd.Series) -> pd.Series:
    rets = df.pct_change().shift(-1).fillna(0.0)
    return rets.loc[~rets.index.year.isin(constants.SKIP_YEARS)]    


def get_position(indices: t.List[pd.Timestamp], n: int = 1, positions: t.Dict[pd.Timestamp, int] = dict()) -> pd.Series:

    if len(indices) == len(positions):
        return pd.Series(positions)

    try:
        date = indices[n-1]
    except IndexError:
        print(f"{n = }")
        print(f"{len(indices) = }")
        print(f"{len(positions) = }")
        raise IndexError
    
    if constants.START["month"] < date.month < constants.END["month"]:
        positions[date] = 1
    
    elif date.month == constants.START["month"]:
        if date.day >= constants.START["day"]:
            positions[date] = 1
        else:
            positions[date] = 0

    elif date.month == constants.END["month"]:
        if date.day <= constants.END["day"]:
            positions[date] = 1
        elif indices[n-2].day <= constants.END["day"]:
            positions[date] = 1
        else:
            positions[date] = 0

    else:
        positions[date] = 0

    assert len(positions) == n, f"{len(positions) = } vs {n = }"
    
    return get_position(indices=indices, n=n+1, positions=positions)


def get_trade_return(returns: pd.Series, pos: pd.Series) -> pd.Series:
    df = pd.concat([returns.to_frame(name="return"), pos.to_frame(name="position")], axis=1)
    df.index.name = "Date"
    
    df["delta_change"] = df["position"].diff()
    df["delta_change"] = df["delta_change"].clip(lower=0.0)
    df["trade"] = df["delta_change"].cumsum()
    
    trade_returns = (
        df.loc[df["trade"] > 0]
        .groupby(by="trade")
        .apply(lambda x: (x["return"] * x["position"] + 1).prod() - 1, include_groups=False)
    )
    date_map = (
        df
        .reset_index()
        .groupby("trade")
        .first()
        ["Date"]
        .to_dict()
    )
    trade_returns = trade_returns.rename(index=date_map)

    return trade_returns


def describe_split(returns: pd.Series, pos: pd.Series = None, 
                   train_dates: t.List = None, valid_dates: t.List = None, test_dates: t.List = None, 
                   **kwargs) -> pd.Series:

    ####
    # Training
    
    train_returns = get_training(df=returns) if train_dates is None else returns.loc[returns.index.isin(train_dates)]
    train_pos = None
    if pos is not None:
        train_pos = get_training(df=pos) if train_dates is None else pos.loc[pos.index.isin(train_dates)]

    train_df = describe(returns=train_returns, pos=train_pos, **kwargs)
    train_df.index = pd.MultiIndex.from_product([["Train"], train_df.index.tolist()], names=['set', 'statistic'])

    ####
    # Validation
    
    valid_returns = get_validation(df=returns) if valid_dates is None else returns.loc[returns.index.isin(valid_dates)]
    valid_pos = None
    if pos is not None:
        valid_pos = get_validation(df=pos) if valid_dates is None else pos.loc[pos.index.isin(valid_dates)]

    valid_df = describe(returns=valid_returns, pos=valid_pos, **kwargs)
    valid_df.index = pd.MultiIndex.from_product([["Validation"], valid_df.index.tolist()], names=['set', 'statistic'])
    
    ####
    # Test
    
    test_returns = get_test(df=returns) if test_dates is None else returns.loc[returns.index.isin(test_dates)]
    test_pos = None
    if pos is not None:
        test_pos = get_test(df=pos) if test_dates is None else pos.loc[pos.index.isin(test_dates)]
    
    test_df = describe(returns=test_returns, pos=test_pos, **kwargs)
    test_df.index = pd.MultiIndex.from_product([["Test"], test_df.index.tolist()], names=['set', 'statistic'])
    
    return pd.concat([train_df, valid_df, test_df], axis=0)


def plot(rets: pd.Series = None, asset: str = None, reset_index: bool = False, **returns: pd.Series):

    asset = asset or "Strategy"
    if rets is not None:
        returns = {asset: rets}

    fig, ax = plt.subplots(nrows=2, sharex="col", figsize=(15, 7))
    twinx = ax[0].twinx()

    for i, (label, ret) in enumerate(returns.items()):
        
        cum_ret = utils.get_cumulative_return(returns=ret, total=False)
        drawdown_statistics = utils.get_drawdown_statistics(returns=ret)
        drawdown = drawdown_statistics.pop("drawdown")

        if reset_index:
            cum_ret = cum_ret.reset_index()[0]
            drawdown = drawdown.reset_index()[0]

        if i == 0:
            ax[0].plot(cum_ret, label=label)
            ax[0].plot(cum_ret.cummax())
        else:
            twinx.plot(cum_ret, label=label)
            twinx.plot(cum_ret.cummax())            
    
        ax[1].plot(-drawdown, label=label)
        
    ax[0].legend(loc="upper left")
    twinx.legend(loc="lower right")
    ax[0].set_title(f"Strategy Return: {asset!r}")
    ax[0].set_ylabel("Cumulative Return (%)")
    ax[0].set_xlabel("Date")

    ax[1].axhline(y=0, label="Breakeven", color="gold")
    ax[1].legend(loc="best")
    ax[1].set_title(f"Strategy Drawdown: {asset!r}")
    ax[1].set_ylabel("Drawdown Percentage (%)")
    ax[1].set_xlabel("Date")

    plt.show()

    return


def get_statistic(stat_df: pd.DataFrame, stat: str) -> pd.DataFrame:
    try:
        return stat_df.loc[stat_df.index.get_level_values("statistic") == stat]
    except KeyError:
        return stat_df.loc[[stat]]


def combine_descriptions(**descriptions) -> pd.DataFrame:
    description_list = []
    for key, desc in descriptions.items():
        desc = desc.copy()

        if constants.SPLIT:
            INDEX0 = list(OrderedDict.fromkeys(list(desc.index.get_level_values(0))))
            INDEX1 = list(OrderedDict.fromkeys(list(desc.index.get_level_values(1))))
            
            print(desc)
            desc.index = pd.MultiIndex.from_product(
                [[key], INDEX0, INDEX1], names=["Type", desc.index.names[0], desc.index.names[1]])
            description_list.append(desc)
        else:
            INDEX0 = list(OrderedDict.fromkeys(list(desc.index.get_level_values(0))))

            desc.index = pd.MultiIndex.from_product(
                [[key], INDEX0], names=["Type", "statistic"])
            description_list.append(desc)
            
    return pd.concat(description_list, axis=0)


def get_strategy_returns(ticker: str, start: dt.date = None, end: dt.date = None) -> t.Dict[str, pd.Series]:

    # Get data from yahoo
    df = utils.get_data(ticker=ticker, columns="Adj Close")

    # Filter data
    if start:
        df = df.loc[df.index.date >= start]
    if end:
        df = df.loc[df.index.date <= end]

    # Calculate returns
    df_ret = get_returns(df=df)

    # Get Positions
    df_pos = get_position(indices=df_ret.index.tolist(), n=1, positions=dict())

    # Get Strategy Returns
    df_ret_strat = df_ret * df_pos
    
    # Get Trade Returns
    df_ret_trade_strat = get_trade_return(returns=df_ret, pos=df_pos)

    return {"returns": df_ret_strat, "pos": df_pos, "trade_returns": df_ret_trade_strat, "hodl": df_ret}


def run2(ticker: str, split: bool = None, details: t.Dict[str, pd.Series] = None, do_plot: bool = False,
       start: dt.date = None, end: dt.date = None) -> None:

    split = constants.SPLIT if split is None else split
    start = constants.START_DATE if start is None else start
    end = constants.END_DATE if end is None else end
    
    strategy_returns = details or get_strategy_returns(ticker=ticker, start=start, end=end)

    # HODL
    df_ret = strategy_returns["hodl"]
    hodl = describe_split(returns=df_ret, daily=True, asset=ticker) if split else utils.describe(returns=df_ret, daily=True, asset=ticker)

    train_dates = utils.get_training(df=df_ret).index.tolist()
    valid_dates = utils.get_validation(df=df_ret).index.tolist()
    test_dates = utils.get_test(df=df_ret).index.tolist()
    
    # Daily Strategy
    df_ret_strat = strategy_returns["returns"]
    df_pos = strategy_returns["pos"]
    daily_strat = (
        describe_split(returns=df_ret_strat, pos=df_pos, daily=True, asset=ticker, 
                       train_dates=train_dates, valid_dates=valid_dates, test_dates=test_dates) 
        if split else utils.describe(returns=df_ret_strat, pos=df_pos, daily=True, asset=ticker)
    )
    
    # Trade Strategy
    df_ret_trade_strat = strategy_returns["trade_returns"]
    stat_trade = (
        describe_split(returns=df_ret_trade_strat, daily=False, asset=ticker, 
                       train_dates=train_dates, valid_dates=valid_dates, test_dates=test_dates) 
        if split else utils.describe(returns=df_ret_trade_strat, daily=False, asset=ticker)
    )

    if do_plot:
        plot(**{"HODL": df_ret, "Strategy (Daily)": df_ret_strat, "Strategy (Trade)": df_ret_trade_strat})

    return pd.concat([
        hodl.to_frame(name="HODL"), 
        daily_strat.to_frame(name="Strategy (Daily)"), 
        stat_trade.to_frame(name="Strategy (Trade)")
    ], axis=1)


def calculate_markowitz_weights(returns_df: pd.DataFrame) -> t.Dict[str, float]:

    # Calculate mean returns and covariance matrix
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    
    # Number of assets
    num_assets = len(mean_returns)
    
    # Define the weight variables
    weights = cp.Variable(num_assets)
    
    # Define the objective function (negative mean-variance)
    objective = cp.Maximize(mean_returns.values @ weights - 0.5 * cp.quad_form(weights, cov_matrix.values))
    
    # Define the constraints (sum of weights equals 1)
    constraints = [cp.sum(weights) == 1, weights >= 0]
    
    # Formulate the problem
    problem = cp.Problem(objective, constraints)
    
    # Solve the problem
    problem.solve()

    return pd.Series(weights.value, index=mean_returns.index).to_dict()
    
