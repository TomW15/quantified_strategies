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
STRATEGY_ID = "strategy_id"
TRADE_ID = "trade_id"



def get_raw_data(assets: str | t.List[str], file: str, is_classification: bool = True, start: dt.date = None, end: dt.date = None) -> t.Tuple[pd.DataFrame, pd.DataFrame]:

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

        if is_classification:
            return (return_data > 0).astype(int)
        
        return return_data

    def get_X() -> pd.DataFrame:
        strategy_returns = pd.read_csv(f"../outputs/{file}", index_col=0, header=[0, 1, 2])
        # strategy_returns = pd.read_csv(f"../outputs/strategy_returns.csv", index_col=0, header=[0, 1, 2])
        
        if DAY_AFTER:
            strategy_returns.loc[:, strategy_returns.columns.get_level_values(1).isin(DAY_AFTER_STRATEGIES)] = (
                strategy_returns.loc[:, strategy_returns.columns.get_level_values(1).isin(DAY_AFTER_STRATEGIES)].shift(1))
            # print(f"Remove {DAY_AFTER_STRATEGIES}")
            # strategy_returns = strategy_returns.loc[:, ~strategy_returns.columns.get_level_values(1).isin(DAY_AFTER_STRATEGIES)]
        
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


def group_trades(X: pd.DataFrame, y: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.DataFrame, t.Dict[str, pd.DataFrame]]:

    assets = y.columns.tolist()
    new_X = X.copy()
    new_y = y.copy()
    
    # Fetch trade id i.e. change in strategy activation
    new_X[STRATEGY_ID] = new_X.apply(lambda x: sum([v * 10 ** i for i, v in enumerate(x.values)]), axis=1)
    new_X[TRADE_ID] = (new_X[STRATEGY_ID].diff().abs().fillna(1.0) > 0).cumsum()
    
    # Copy trade id to target/asset return dataframe
    new_y[TRADE_ID] = new_X[TRADE_ID]
    
    # Find trade id to first trade date
    trades = new_y.reset_index().set_index(TRADE_ID)["Date"]
    trades = trades[~trades.index.duplicated()]
    trades_to_date_dict = trades.to_dict()
    
    # Find return for each trade id
    new_y = new_y.groupby(by=TRADE_ID)[assets].apply(lambda ret: strategy_utils.get_cumulative_return(returns=ret, total=True))
    new_y.index = new_y.index.map(trades_to_date_dict)
    new_y.index.name = "Date"

    X = X.loc[X.index.isin(new_y.index)].copy()

    # Calculate holding period before change
    X = pd.concat([X, ], axis=1)
    
    # Add add last position columns
    X = pd.concat([
        X, 
        pd.DataFrame(0, index=X.index, columns=[f"pos_{asset}" for asset in y.columns]),
    ], axis=1)

    hold_days = pd.Series((X.index[1:] - X.index[:-1]).days.tolist() + [1], index=X.index, name=HOLD_DAYS)

    return X, new_y, {HOLD_DAYS: hold_days}



def get_data(assets: str | t.List[str], file: str, start: dt.date = None, end: dt.date = None):

    # Get raw data i.e. X and y for each day
    X, y = get_raw_data(assets=assets, file=file, is_classification=False)

    if start is not None:
        X = X.loc[X.index.date >= start]
        y = y.loc[y.index.date >= start]

    if end is not None:
        X = X.loc[X.index.date <= end]
        y = y.loc[y.index.date <= end]

    # Group data
    X_grouped, y_grouped, additional = group_trades(X=X, y=y)

    return X_grouped, y_grouped, additional


def run(net: nn.Module, X_df: pd.DataFrame, y_df: pd.DataFrame, **kwargs) -> t.Dict[str, pd.DataFrame]:
    
    def get_allocation():
    
        def _get_allocation(index: t.Any, last_pos: pd.Series):
            x = pd.concat([X_df.loc[[index]], last_pos.to_frame(name=index).T], axis=1)
            x_tensor, _ = ml_utils.convert_data_to_tensors(X=x, y=y_df)
            new_pos = pd.Series(net(x_tensor).reshape(-1).detach().numpy(), index=y_df.columns)
            return new_pos
        
        A = []
        last_position = pd.Series(0.0, index=y_df.columns)
        hold_days = 0
        last_X_df = X_df.loc[X_df.index[-1]]
        for index in X_df.index:
            if all(last_X_df == X_df.loc[index]):
                pass
            else:
                last_position = _get_allocation(index=index, last_pos=last_position)
            A.append(last_position.to_frame(name=index).T)
            last_X_df = X_df.loc[index].copy()
            
        A = pd.concat(A, axis=0)

        return A

    assets = y_df.columns.tolist()
    
    long_costs_df = pd.Series(kwargs.get("long_costs", 0.0), index=assets)
    short_costs_df = pd.Series(kwargs.get("short_costs", 0.0), index=assets)
    
    # Get Allocation
    alloc = get_allocation()
    
    benchmark_alloc = X_df.T.groupby(level=2).sum().T
    benchmark_alloc = benchmark_alloc.div(benchmark_alloc.sum(axis=1), axis=0).fillna(0.0)
    benchmark_alloc = benchmark_alloc.reindex(columns=y_df.columns, fill_value=0.0)
    benchmark_alloc[CASH] = 1 - benchmark_alloc.sum(axis=1)
    
    hodl_alloc = pd.DataFrame(1 / (len(assets) - 1), index=alloc.index, columns=assets)
    hodl_alloc[CASH] = 0.0

    # Get Strategy returns
    hodl_ret = (y_df * hodl_alloc).sum(axis=1) - (hodl_alloc.clip(lower=0.0).dot(long_costs_df) + hodl_alloc.clip(upper=0.0).abs().dot(short_costs_df))
    strat_bm_ret = (y_df * benchmark_alloc).sum(axis=1) - (benchmark_alloc.clip(lower=0.0).dot(long_costs_df) + benchmark_alloc.clip(upper=0.0).abs().dot(short_costs_df))
    strat_ret = (y_df * alloc).sum(axis=1) - (alloc.clip(lower=0.0).dot(long_costs_df) + alloc.clip(upper=0.0).abs().dot(short_costs_df))

    
    def format_alloc(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={col: f"w_{col}" for col in df.columns})
    
    df_strat = pd.concat([y_df, format_alloc(df=alloc), strat_ret.to_frame(name="ret")], axis=1)
    df_benchmark = pd.concat([y_df, format_alloc(df=benchmark_alloc), strat_bm_ret.to_frame(name="ret")], axis=1)
    df_hodl = pd.concat([y_df, format_alloc(df=hodl_alloc), hodl_ret.to_frame(name="ret")], axis=1)

    return {"strat": df_strat, "benchmark": df_benchmark, "hodl": df_hodl}



