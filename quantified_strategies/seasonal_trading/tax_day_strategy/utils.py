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

import constants


def get_data(ticker: str) -> pd.Series:
    with warnings.catch_warnings(action="ignore"):
        data = yf.download(tickers=ticker, progress=False)["Adj Close"]
        return data


def get_returns(df: pd.Series) -> pd.Series:
    rets = df.pct_change().fillna(0.0)
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


def get_cumulative_return(returns: pd.Series, total: bool = False) -> pd.Series | float:
    if total:
        return (returns + 1).prod() - 1
    return (returns + 1).cumprod() - 1

    
def get_drawdown_statistics(returns: pd.Series) -> t.Dict[str, t.Any]:

    cum_ret = get_cumulative_return(returns=returns, total=False)

    # Drawdown
    drawdown = -(cum_ret - cum_ret.cummax()) / (cum_ret.cummax() + 1)
    # Max Drawdown
    max_drawdown = drawdown.max()
    # Max Drawdown Date
    max_drawdown_bottom = drawdown.idxmax()
    # Max Drawdown Start Date
    max_drawdown_top = cum_ret[:max_drawdown_bottom].idxmax()
    # Max Drawdown Duration
    max_drawdown_duration = pd.Timedelta(days=cum_ret[max_drawdown_top:max_drawdown_bottom].shape[0])
    # Max Drawdown Recovery Date
    try:
        max_drawdown_recovery = cum_ret[(cum_ret >= (cum_ret[max_drawdown_top] + 1e-8)) & (cum_ret.index > max_drawdown_top)].index[0]
    except IndexError:
        max_drawdown_recovery = None
    # Max Drawdown Recovery Duration
    max_drawdown_recovery_duration = None if max_drawdown_recovery is None else pd.Timedelta(days=cum_ret[max_drawdown_bottom:max_drawdown_recovery].shape[0])

    drawdown_duration = {
        "drawdown": drawdown,
        "max_drawdown": max_drawdown,
        "max_drawdown_bottom": max_drawdown_bottom.date(),
        "max_drawdown_top": max_drawdown_top.date(),
        "max_drawdown_duration": max_drawdown_duration.days,
        "max_drawdown_recovery": max_drawdown_recovery if max_drawdown_recovery is None else max_drawdown_recovery.date(),
        "max_drawdown_recovery_duration": max_drawdown_recovery_duration if max_drawdown_recovery_duration is None else max_drawdown_recovery_duration.days,
    }

    return drawdown_duration


def describe(returns: pd.Series, pos: pd.Series = None, daily: bool = True, asset: str = None) -> pd.Series:

    asset = asset or "Undefined"
    
    if pos is None:
        pos = pd.Series(1, index=returns.index, dtype=int)

    active_returns = returns[pos != 0.0]

    # Activity
    activity_ratio = active_returns.shape[0] / returns.shape[0]
    number_of_trades = pos.diff().clip(lower=0.0).sum()
    average_hold_period = activity_ratio * returns.shape[0] / (number_of_trades or 1)
    
    # Return Statistics
    mu_ret = active_returns.mean()
    med_ret = active_returns.median()
    std_ret = active_returns.std()
    tot_ret = get_cumulative_return(returns=active_returns, total=True)
    cum_ret = get_cumulative_return(returns=active_returns, total=False)
    ret_q1 = active_returns[active_returns < med_ret].median()
    ret_q3 = active_returns[active_returns > med_ret].median()
    cagr = ((tot_ret + 1) / 1.0) ** (1 / active_returns.shape[0]) - 1
    trade_cagr = (1 + cagr) ** average_hold_period - 1
    ann_cagr = (1 + cagr) ** 252 - 1

    # Drawdown
    drawdown_statistics = get_drawdown_statistics(returns=active_returns)
    drawdown = drawdown_statistics.pop("drawdown")
    
    # Return Ratios
    sharpe_ratio = mu_ret / std_ret
    ann_sharpe_ratio = sharpe_ratio * np.sqrt(252 * activity_ratio)
    sortino_ratio = mu_ret /  active_returns[active_returns <= 0.0].std()
    ann_sortino_ratio = sortino_ratio * np.sqrt(252 * activity_ratio)
    calmar_ratio = tot_ret / drawdown_statistics["max_drawdown"]
    hit_ratio = (active_returns>0).mean()
    profit_factor = -active_returns[active_returns > 0].sum() / active_returns[active_returns < 0].sum()
    
    statistics = {
        "Asset": asset,

        # Strategy Start-End
        "Start": returns.index[0].date(),
        "End": returns.index[-1].date(),
        
        # Return Statistics
        "Mean Return": f"{mu_ret:,.5%}",
        "Total Return": f"{tot_ret:,.2%}", 
        "Median Return": f"{med_ret:,.5%}",
        "1st Quartile": f"{ret_q1:,.5%}",
        "3rd Quartile": f"{ret_q3:,.5%}",
        "Std Dev Return": f"{std_ret:,.5%}",
        "CAGR": f"{cagr * 10_000:.3f} bps",
        "Trade CAGR": f"{trade_cagr:,.3%}" if any(pos == 0) else "N/A",
        "Ann. CAGR": f"{ann_cagr:,.3%}" if daily else "N/A",

        # Activity Ratio
        "Activity Ratio": f"{activity_ratio:.2%}",
        "Number of Trades": number_of_trades,
        "Average Hold Period": f"{average_hold_period:,.2f} Days",
        
        # Return Ratios
        "Daily Sharpe Ratio": round(sharpe_ratio, 4),
        "Ann. Sharpe Ratio": round(ann_sharpe_ratio, 4),
        "Daily Sortino Ratio": round(sortino_ratio, 4),
        "Ann. Sortino Ratio": round(ann_sortino_ratio, 4),   
        "Daily Calmar Ratio": round(calmar_ratio, 4),
        "Hit Ratio": f"{hit_ratio:.2%}",
        "Profit Factor": f"{profit_factor:.2f}x",

        # Drawdown
        "MDD": f"{-drawdown_statistics['max_drawdown']:.2%}",
        "MDD Start": drawdown_statistics["max_drawdown_top"],
        "MDD Bottom": drawdown_statistics["max_drawdown_bottom"],
        "MDD End": drawdown_statistics["max_drawdown_recovery"],
        "MDD Decline Duration": None if drawdown_statistics['max_drawdown_duration'] is None else f"{drawdown_statistics['max_drawdown_duration']} Days",
        "MDD Recovery Duration": None if drawdown_statistics['max_drawdown_recovery_duration'] is None else f"{drawdown_statistics['max_drawdown_recovery_duration']} Days",
        
    }

    return pd.Series(statistics, dtype=object)


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


def plot(rets: pd.Series = None, asset: str = None, **returns: pd.Series):

    asset = asset or "Strategy"
    if rets is not None:
        returns = {asset: rets}

    fig, ax = plt.subplots(nrows=2, sharex="col", figsize=(15, 7))
    twinx = ax[0].twinx()

    for i, (label, ret) in enumerate(returns.items()):
        
        cum_ret = get_cumulative_return(returns=ret, total=False)
        drawdown_statistics = get_drawdown_statistics(returns=ret)
        drawdown = drawdown_statistics.pop("drawdown")

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


def get_training(df: pd.Series) -> pd.Series:
    df_train, _ = train_test_split(df, train_size=0.8, shuffle=False)
    df_train, _ = train_test_split(df_train, train_size=0.5/0.8, shuffle=False)
    return df_train


def get_validation(df: pd.Series) -> pd.Series:
    df_train, _ = train_test_split(df, train_size=0.8, shuffle=False)
    _, df_valid = train_test_split(df_train, train_size=0.5/0.8, shuffle=False)
    return df_valid


def get_test(df: pd.Series) -> pd.Series:  
    _, df_test = train_test_split(df, train_size=0.8, shuffle=False)
    return df_test


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


def get_strategy_returns(ticker: str) -> t.Dict[str, pd.Series]:

    # Get data from yahoo
    df = get_data(ticker=ticker)

    # Calculate returns
    df_ret = get_returns(df=df)

    # Get Positions
    df_pos = get_position(indices=df_ret.index.tolist(), n=1, positions=dict())

    # Get Strategy Returns
    df_ret_strat = df_ret * df_pos
    
    # Get Trade Returns
    df_ret_trade_strat = get_trade_return(returns=df_ret, pos=df_pos)

    return {"returns": df_ret_strat, "pos": df_pos, "trade_returns": df_ret_trade_strat, "hodl": df_ret}


def run(ticker: str, split: bool = None, details: t.Dict[str, pd.Series] = None, do_plot: bool = False) -> None:

    split = constants.SPLIT if split is None else split
    
    strategy_returns = details or get_strategy_returns(ticker=ticker)

    # HODL
    df_ret = strategy_returns["hodl"]
    hodl = describe_split(returns=df_ret, daily=True, asset=ticker) if split else describe(returns=df_ret, daily=True, asset=ticker)

    train_dates = get_training(df=df_ret).index.tolist()
    valid_dates = get_validation(df=df_ret).index.tolist()
    test_dates = get_test(df=df_ret).index.tolist()
    
    # Daily Strategy
    df_ret_strat = strategy_returns["returns"]
    df_pos = strategy_returns["pos"]
    daily_strat = (
        describe_split(returns=df_ret_strat, pos=df_pos, daily=True, asset=ticker, 
                       train_dates=train_dates, valid_dates=valid_dates, test_dates=test_dates) 
        if split else describe(returns=df_ret_strat, pos=df_pos, daily=True, asset=ticker)
    )
    
    # Trade Strategy
    df_ret_trade_strat = strategy_returns["trade_returns"]
    stat_trade = (
        describe_split(returns=df_ret_trade_strat, daily=False, asset=ticker, 
                       train_dates=train_dates, valid_dates=valid_dates, test_dates=test_dates) 
        if split else describe(returns=df_ret_trade_strat, daily=False, asset=ticker)
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
    
