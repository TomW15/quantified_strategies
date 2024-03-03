# packages

import numpy as np
import pandas as pd
import typing as t

 
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


