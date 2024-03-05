# packages

import cvxpy as cp
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import typing as t
import warnings
import yfinance as yf


def get_data(ticker: str, columns: str | t.List[str] = None, flt: bool = False, source: str = "eod", **kwargs) -> pd.DataFrame | pd.Series:
    with warnings.catch_warnings(action="ignore"):
        
        if source == "yahoo":
            data = yf.download(tickers=ticker, progress=False, **kwargs)
            
        elif source == "eod":
            
            print(kwargs)
            import requests
            print("remove api from here")
            url = f"https://eodhd.com/api/eod/{ticker}?api_token=65a9557c3df693.14557024&fmt=json"
            data = requests.get(url).json()
            data = pd.DataFrame(data)
            data = data.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "adjusted_close": "Adj Close"})
            data = data.set_index("Date")
            data.index = pd.DatetimeIndex(data.index)

        if flt:
            data = data.loc[data["Adj Close"].pct_change().abs() < 4 * data["Adj Close"].pct_change().std()]
        if columns is None:
            return data
        data = data[columns]
        return data


def get_today_data(ticker: str, columns: str | t.List[str] = None) -> pd.DataFrame | pd.Series:
    with warnings.catch_warnings(action="ignore"):
        data = yf.download(
            tickers=ticker, 
            progress=False, 
            interval="1m", 
            start=dt.date.today() - dt.timedelta(days=1), 
            end=dt.date.today() + dt.timedelta(days=1),
        )
        data = data.loc[data.index.date == dt.date.today()]
        if columns is None:
            return data
        data = data[columns]
        return data


def get_intraday_data(ticker: str, columns: str | t.List[str] = None, start: dt.date = None, end: dt.date = None) -> pd.DataFrame | pd.Series:

    import requests
    print("remove api from here")
    url = f"https://eodhd.com/api/intraday/{ticker}?api_token=65a9557c3df693.14557024&fmt=json&interval=1m"
    data = requests.get(url).json()
    data = pd.DataFrame(data)
    data = data.rename(columns={"datetime": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close"})
    data = data.set_index("Date")
    data.index = pd.DatetimeIndex(data.index)

    if start is not None:
        data = data.loc[data.index.date >= start]
    if end is not None:
        data = data.loc[data.index.date <= end]
    
    if columns is None:
        return data
    data = data[columns]
    return data

 
def get_cumulative_return(returns: pd.Series, total: bool = False) -> pd.Series | float:
    if total:
        return (returns + 1).prod() - 1
    return (returns + 1).cumprod() - 1


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
    # activity_ratio = active_returns.shape[0] / returns.shape[0]
    activity_ratio = pos.sum() / pos.shape[0]
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
    ann_sharpe_ratio = sharpe_ratio * np.sqrt(252)
    ann_adj_sharpe_ratio = sharpe_ratio * np.sqrt(252 * activity_ratio)
    sortino_ratio = mu_ret /  active_returns[active_returns <= 0.0].std()
    ann_sortino_ratio = sortino_ratio * np.sqrt(252)
    ann_adj_sortino_ratio = sortino_ratio * np.sqrt(252 * activity_ratio)
    calmar_ratio = tot_ret / drawdown_statistics["max_drawdown"]
    hit_ratio = (active_returns > 0).mean()
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
        "Trade CAGR": f"{trade_cagr:,.3%}" if daily else "N/A",
        "Ann. CAGR": f"{ann_cagr:,.3%}" if daily else "N/A",

        # Activity Ratio
        "Activity Ratio": f"{activity_ratio:.2%}",
        "Number of Trades": number_of_trades,
        "Average Hold Period": f"{average_hold_period:,.2f} Days",
        
        # Return Ratios
        "Daily Sharpe Ratio": round(sharpe_ratio, 4),
        "Ann. Sharpe Ratio": round(ann_sharpe_ratio, 4) if daily else "N/A",
        "Adj. Ann. Sharpe Ratio": round(ann_adj_sharpe_ratio, 4) if daily else "N/A",
        "Daily Sortino Ratio": round(sortino_ratio, 4),
        "Ann. Sortino Ratio": round(ann_sortino_ratio, 4) if daily else "N/A",  
        "Adj. Ann. Sortino Ratio": round(ann_adj_sortino_ratio, 4) if daily else "N/A", 
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


def calculate_markowitz_weights(returns_df: pd.DataFrame) -> t.Dict[str, float]:

    # To force a observation every day
    returns_df = returns_df.fillna(0.0)
    
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

    return pd.Series(weights.value, index=mean_returns.index)
