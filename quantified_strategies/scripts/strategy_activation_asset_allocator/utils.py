import datetime as dt
import pandas as pd
from torch import nn
import typing as t

from quantified_strategies import ml_utils, strategy_utils, utils

CASH = "CASH"
ENTRY = "Adj Close"
DAY_AFTER = True
DAY_AFTER_STRATEGIES = ["buy_when_yields_are_low"]
HOLD_DAYS = "hold_days"
RISK_FREE_RATE = 0.0
FILE = "strategy_returns_alternative.csv"


def get_raw_data(assets: str | t.List[str], file: str, start: dt.date = None, end: dt.date = None) -> t.Tuple[pd.DataFrame, pd.DataFrame]:

    def get_y() -> pd.DataFrame:
        
        # ENTRY = "Adj Close"
        price_data = [strategy_utils.get_data(ticker=ticker, columns=ENTRY).to_frame(name=ticker) for ticker in assets if ticker != CASH]
        price_data = pd.concat(price_data, axis=1)
        return_data = price_data.pct_change()
        
        if CASH in assets:
            # risk_free_rate = strategy_utils.get_data(ticker="^TNX", columns=ENTRY, source="yahoo").to_frame(name=CASH)
            risk_free_rate = pd.DataFrame(RISK_FREE_RATE, index=return_data.index, columns=[CASH])
            risk_free_rate = (1 + risk_free_rate / 100) ** (1 / 252) - 1
            return_data[CASH] = risk_free_rate.reindex(index=return_data.index, method="ffill").bfill()
            
        return_data = return_data.shift(-1)

        return_data = return_data.dropna()
        
        return return_data

    def get_X() -> pd.DataFrame:
        strategy_returns = pd.read_csv(f"../outputs/{file}", index_col=0, header=[0, 1, 2])
        # strategy_returns = pd.read_csv(f"../outputs/strategy_returns.csv", index_col=0, header=[0, 1, 2])
        
        if DAY_AFTER:
            strategy_returns.loc[:, strategy_returns.columns.get_level_values(1).isin(DAY_AFTER_STRATEGIES)] = (
                strategy_returns.loc[:, strategy_returns.columns.get_level_values(1).isin(DAY_AFTER_STRATEGIES)].shift(1))
            strategy_returns = strategy_returns.loc[:, ~strategy_returns.columns.get_level_values(1).isin(DAY_AFTER_STRATEGIES)]
        
        strategy_returns = strategy_returns.loc[:, strategy_returns.columns.get_level_values(2).isin(assets)]
        strategy_returns.index = pd.DatetimeIndex(strategy_returns.index)
        is_active = ~(strategy_returns.isna())
        is_active = is_active.astype(int)

        return is_active

    assets = assets if isinstance(assets, list) else [assets]

    # Get target variables: these are the returns from entering a position from close to close t+1
    y = get_y()
    
    # Get explanatory variables: these are the signals from the strategies indicating whether to buy or not
    X = get_X()

    X = X.loc[X.index.isin(X.index.intersection(y.index))]
    y = y.loc[y.index.isin(y.index.intersection(X.index))]

    X = X.sort_index()
    y = y.sort_index()

    if start is not None:
        X = X.loc[X.index.date >= start]
        y = y.loc[y.index.date >= start]

    if end is not None:
        X = X.loc[X.index.date <= end]
        y = y.loc[y.index.date <= end]

    return X, y


def run(X_df: pd.DataFrame, y_df: pd.DataFrame, **kwargs) -> t.Dict[str, pd.DataFrame]:
    
    assets = y_df.columns.tolist()
    
    long_costs_df = pd.Series(kwargs.get("long_costs", 0.0), index=assets)
    short_costs_df = pd.Series(kwargs.get("short_costs", 0.0), index=assets)

    # Allocation
    ## Strategy Activation
    alloc = X_df.T.groupby(level=2).sum().T
    alloc = alloc.div(benchmark_alloc.sum(axis=1), axis=0).fillna(0.0)
    alloc = alloc.reindex(columns=y_df.columns, fill_value=0.0)
    alloc[CASH] = 1 - alloc.sum(axis=1)

    ## HODL
    hodl_alloc = pd.DataFrame(1 / (len(assets) - 1), index=alloc.index, columns=assets)
    hodl_alloc[CASH] = 0.0

    # Get Strategy returns
    strat_ret = (y_df * alloc).sum(axis=1) - (alloc.clip(lower=0.0).dot(long_costs_df) + alloc.clip(upper=0.0).abs().dot(short_costs_df))
    hodl_ret = (y_df * hodl_alloc).sum(axis=1) - (hodl_alloc.clip(lower=0.0).dot(long_costs_df) + hodl_alloc.clip(upper=0.0).abs().dot(short_costs_df))

    def format_alloc(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={col: f"w_{col}" for col in df.columns})
    
    df_strat = pd.concat([y_df, format_alloc(df=alloc), strat_ret.to_frame(name="ret")], axis=1)
    df_hodl = pd.concat([y_df, format_alloc(df=hodl_alloc), hodl_ret.to_frame(name="ret")], axis=1)

    return {"strat": df_strat, "hodl": df_hodl}
