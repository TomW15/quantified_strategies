import torch
from torch import nn
import typing as t

BORROWING_COSTS: str = "borrowing_costs"


def my_sharpe_loss(output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    
    port_return = output * target
    try:
        port_return = torch.sum(port_return, dim=1)
    except IndexError:
        pass
    
    mean_return = torch.mean(port_return)
    std_return = torch.std(port_return)
    
    sharpe = mean_return / (std_return + 1e-10)

    num = torch.sum(torch.abs(output))
    denom = output.shape[0]
    multiplier = torch.sqrt(252 * num / denom)
    
    ann_sharpe = multiplier * sharpe
    
    return ann_sharpe


def my_return_loss(weights: torch.Tensor, returns: torch.Tensor, **kwargs) -> torch.Tensor:
    
    port_return = weights * returns
    try:
        port_return = torch.sum(port_return, dim=1)
    except IndexError:
        pass
    
    total_return = torch.prod(port_return + 1) - 1
    
    return total_return


def my_cagr_loss(weights: torch.Tensor, returns: torch.Tensor, **kwargs) -> torch.Tensor:


    port_return = weights * returns
    try:
        port_return = torch.sum(port_return, dim=1)
    except IndexError:
        pass
        
    borrowing_costs = calc_borrowing_cost(weights=weights, borrowing_costs=kwargs.get(BORROWING_COSTS))
    
    port_return = port_return - borrowing_costs
    total_return = torch.prod(port_return + 1) - 1
    
    initial_value = 1.0
    activity = weights.shape[0]
    
    cagr = ((total_return + initial_value) / initial_value) ** (1 / activity) - 1    
    cagr_bps = cagr * 10_000
    
    return cagr_bps


def calc_long_overnight_cost(weights: torch.Tensor, long_costs: t.List[float] = None, n_days: t.List[int] = None) -> torch.Tensor:

    if long_costs is None:
        long_costs = [0.0 for _ in range(weights.shape[1])]

    if n_days is None:
        n_days = [1 for _ in range(weights.shape[0])]
    
    long_weights = nn.Threshold(threshold=0.0, value=0.0)(weights)
    
    costs = torch.mul(long_weights, torch.Tensor(long_costs))
    total_cost = torch.sum(costs, dim=1)

    total_cost = total_cost * torch.Tensor(n_days)
    
    return total_cost


def calc_short_overnight_cost(weights: torch.Tensor, short_costs: t.List[float] = None, n_days: t.List[int] = None) -> torch.Tensor:

    if short_costs is None:
        short_costs = [0.0 for _ in range(weights.shape[1])]
        
    if n_days is None:
        n_days = [1 for _ in range(weights.shape[0])]
    
    short_weights = torch.abs(weights - nn.Threshold(threshold=0.0, value=0.0)(weights))
    
    costs = torch.mul(short_weights, torch.Tensor(short_costs))
    total_cost = torch.sum(costs, dim=1)

    total_cost = total_cost * torch.Tensor(n_days)
    
    return total_cost


def calc_borrowing_cost(weights: torch.Tensor, borrowing_costs: t.List[float] = None, n_days: t.List[int] = None) -> torch.Tensor:

    if borrowing_costs is None:
        borrowing_costs = [0.0 for _ in range(weights.shape[1])]
        
    if n_days is None:
        n_days = [1 for _ in range(weights.shape[0])]
    
    borrowed_weights = torch.abs(weights - nn.Threshold(threshold=0.0, value=0.0)(weights))
    
    costs = torch.mul(borrowed_weights, torch.Tensor(borrowing_costs))
    total_cost = torch.sum(costs, dim=1)
    
    total_cost = total_cost * torch.Tensor(n_days)
    
    return total_cost
