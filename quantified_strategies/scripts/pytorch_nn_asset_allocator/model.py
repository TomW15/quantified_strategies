
from loguru import logger
import numpy as np
import os
import pandas as pd
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import time
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import typing as t
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from quantified_strategies import loss, ml_utils, plot_utils, strategy_utils, utils

DEVICE = ml_utils.get_device()

BATCH_SIZE: int = 64
EARLY_STOPPING_PATIENCE: int = 10
EARLY_STOPPING_MIN_DELTA: float = 0.0
EARLY_STOPPING_MIN_PERIODS: int = 100 # 0
EPOCHS: int = 2_000
LEARNING_RATE: float = 0.001
PATH: Path = Path(os.getcwd()).parent
SHUFFLE: bool = False
TRAIN_SIZE: float = 0.7


def update_datasets(net: nn.Module, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame
                   ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    def update_dataset(X: pd.DataFrame, y: pd.DataFrame) -> t.Tuple[torch.Tensor, torch.Tensor]:

        # Get positions from network
        # TODO: FIX THIS!
        X_tensor, y_tensor = ml_utils.convert_data_to_tensors(X=X, y=y)
        X_tensor, y_tensor = Net.translate(X=X_tensor, y=y_tensor)
        y_pred_tensor = net(X_tensor)
        y_pred = pd.DataFrame(y_pred_tensor.detach().numpy(), index=X.index, columns=[f"pos_{asset}" for asset in y.columns])
    
        # Combine data
        new_X = pd.concat([X.iloc[:, :-y.shape[1]], y_pred.shift(1).fillna(0.0)], axis=1)
        # Convert dataset
        X, y = ml_utils.convert_data_to_tensors(X=new_X, y=y_tensor)

        assert X.shape[1] == net.input_shape
        assert y.shape[1] == net.output_shape

        return X, y

    net.eval()
    X_train_tensor, y_train_tensor = update_dataset(X=X_train, y=y_train)
    if X_test.shape[0] == 0:
        X_test_tensor, y_test_tensor = torch.Tensor(), torch.Tensor()
    else:
        X_test_tensor, y_test_tensor = update_dataset(X=X_test, y=y_test)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    

def train(net: nn.Module, X: pd.DataFrame, y: pd.DataFrame, hold_days: pd.Series, loss_fn, name: str = None,
          store: bool = False, lr: bool = LEARNING_RATE, batch_size: int = BATCH_SIZE, epochs: int = EPOCHS,
          maximize_loss: bool = False, test_size: float = 1-TRAIN_SIZE, 
          patience: int = EARLY_STOPPING_PATIENCE, min_delta: float = EARLY_STOPPING_MIN_DELTA,
          min_periods: int = EARLY_STOPPING_MIN_PERIODS, verbose: bool = True, **loss_kwargs):

    optimizer = optim.Adam(net.parameters(), lr=lr, maximize=maximize_loss)
    early_stopping_train = ml_utils.EarlyStopping(
        patience=patience, 
        min_delta=min_delta, 
        maximize=maximize_loss,
        min_periods=min_periods,
    )
    early_stopping_test = ml_utils.EarlyStopping(
        patience=patience * 5, 
        min_delta=min_delta, 
        maximize=maximize_loss,
        min_periods=min_periods,
    )

    if name is None:
        MODEL_NAME: str = f"{int(time.time())}"
    else:
        MODEL_NAME: str = name

    if verbose:
        logger.info(f"Training: {MODEL_NAME!r}")

    if test_size:
        X_train, X_test, y_train, y_test = ml_utils.split_data(X=X, y=y, train_size=1-test_size, shuffle=False)
        hold_days_train, hold_days_test = train_test_split(hold_days, train_size=1-test_size, shuffle=False)
    else:
        X_train, X_test, y_train, y_test = X.copy(), pd.DataFrame(columns=X.columns), y.copy(), pd.DataFrame(columns=y.columns)
        hold_days_train, hold_days_test = hold_days, pd.Series(dtype=float, name="hold_days")

    with open(PATH / f"./outputs/models/{net.MODEL_TYPE}-model-{MODEL_NAME}.log", "a") as log:

        for epoch in range(epochs):

            X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = update_datasets(
                net=net, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

            net.train()
            for i in range(0, len(X_train_tensor), batch_size):

                batch_X = X_train_tensor[i:(i + batch_size)].to(device=DEVICE)
                batch_y = y_train_tensor[i:(i + batch_size)].to(device=DEVICE)

                acc, loss = ml_utils.fwd_pass(net=net, loss_fn=loss_fn, X=batch_X, y=batch_y, 
                                              optimizer=optimizer, do_train=True, n_days=hold_days_train.iloc[i:(i + batch_size)].tolist(),
                                              **loss_kwargs)

            net.eval()
            loss, hit_rate = ml_utils.evaluate(net=net, loss_fn=loss_fn, X=X_train_tensor, y=y_train_tensor, n_days=hold_days_train.tolist())
            
            val_loss, val_hit_rate = loss, hit_rate
            if test_size:
                val_loss, val_hit_rate = ml_utils.evaluate(net=net, loss_fn=loss_fn, X=X_test_tensor, y=y_test_tensor, n_days=hold_days_test.tolist())
                
            if store:
                log.write(f"{MODEL_NAME},{time.time():.3f},{epoch},{loss:.6f},{hit_rate:.6f},{val_loss:.6f},{val_hit_rate:.6f}\n")
                
            if verbose and epoch % 50 == 0:
                logger.info(f"Epoch: {epoch} / {epochs}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}," +\
                            f"Hit Rate: {hit_rate:.2%}, Val Hit Rate: {val_hit_rate:.2%}")

            # early stopping
            early_stopping_train(loss)
            early_stopping_test(val_loss)
            if early_stopping_train.early_stop or (early_stopping_test.early_stop and test_size > 0):
                if verbose:
                    logger.info(f"Early Stopping reached @ {epoch = }! Best Loss: {early_stopping_train.best_loss}, Loss: {loss}, " +\
                                f"Best Val Loss: {early_stopping_test.best_loss}, Val Loss: {val_loss}, " +\
                                f"Hit Rate: {hit_rate:.2%}, Val Accuracy: {val_hit_rate:.2%}")
                break

    if store:
        net.save(name=MODEL_NAME)
        shutil.copy(PATH / f"./outputs/models/{net.MODEL_TYPE}-model-{MODEL_NAME}.log", 
                    PATH / f"./outputs/models/{net.MODEL_TYPE}-model-latest.log")
    else:
        os.remove(PATH / f"./outputs/models/{net.MODEL_TYPE}-model-{MODEL_NAME}.log")

    return


def my_cagr_loss(weights: torch.Tensor, returns: torch.Tensor, **kwargs) -> torch.Tensor:


    port_return = weights * returns
    try:
        port_return = torch.sum(port_return, dim=1)
    except IndexError:
        pass

    long_overnight_costs = loss.calc_long_overnight_cost(weights=weights, long_costs=kwargs.get("long_costs"), n_days=kwargs.get("n_days"))
    short_overnight_costs = loss.calc_short_overnight_cost(weights=weights, short_costs=kwargs.get("short_costs"), n_days=kwargs.get("n_days"))
    overnight_costs = long_overnight_costs + short_overnight_costs
        
    port_return = port_return - overnight_costs

    active_weights = 1 - weights[:, -1]
    active_port_return = port_return[torch.abs(active_weights) > 1e-5]
    mu_return = torch.mean(active_port_return)
    
    # thresh = 0.0
    # while active_port_return[active_port_return < thresh].shape[0] <= 3:
    #     thresh += 1 / 10_000
    #     if thresh > 10 / 10_000:
    #         thresh = 5
    #         break
    
    std_return = torch.std(active_port_return) if active_port_return.shape[0] > 3 else 0.01
    
    sharpe = mu_return / (std_return + 1e-8)
    multiplier = torch.sum(active_weights) / (len(active_weights) + 1e-8)
    if multiplier < 0.1:
        ann_sharpe = sharpe * multiplier
    else:
        ann_sharpe = sharpe #* torch.sqrt(multiplier)

    return ann_sharpe

    # total_return = torch.prod(port_return + 1) - 1
    
    # initial_value = 1.0
    # activity = weights.shape[0]
    
    # cagr = ((total_return + initial_value) / initial_value) ** (1 / activity) - 1    
    # cagr_bps = cagr * 10_000
    
    # return cagr_bps


def calc_trading_costs(weights: torch.Tensor, rate: torch.Tensor = None, fixed_costs: t.List[float] = None, var_costs: t.List[float] = None) -> torch.Tensor:

    if fixed_costs is None:
        fixed_costs = [0.0 for _ in range(weights.shape[1])]
    if var_costs is None:
        var_costs = [0.0 for _ in range(weights.shape[1])]
    
    # TODO: incorporate length held
    # NOTE: Requires weights not to be shuffled!
    zeros = torch.zeros((1, weights.shape[1]))
    weights = torch.cat((weights, zeros), dim=0)
    
    weights_to = weights[1:]
    weights_from = weights[:-1]
    
    change_in_weights = weights_to - weights_from
    abs_change_in_weights = torch.abs(change_in_weights)
    has_weights_changed = nn.Threshold(threshold=1e-10, value=1.0)(abs_change_in_weights)

    var_cost = torch.mul(abs_change_in_weights, torch.Tensor(var_costs))
    assert var_cost.shape == abs_change_in_weights.shape

    fixed_cost = torch.mul(has_weights_changed, torch.Tensor(fixed_costs))
    assert fixed_cost.shape == has_weights_changed.shape
    
    trading_cost = var_cost + fixed_cost
    total_trading_cost = torch.sum(trading_cost, dim=1)

    if rate is None:
        return total_trading_cost

    total_trading_cost = torch.mul(total_trading_cost, rate)

    return total_trading_cost



class Net(nn.Module):

    DEFAULT_LAYER_SIZES: t.List[int] = [8, 16, 8]
    DEFAULT_ALLOW_NEGATIVE_WEIGHTS: bool = False

    # Used when more than one asset is being traded, enables leverage.
    DEFAULT_MAX_WEIGHT: float = 1.0
    DEFAULT_MIN_WEIGHT: float = 0.0
    
    # Model Type: used to save model
    MODEL_TYPE: str = "nn-position-memory"
    
    def __init__(self, input_shape: int, output_shape: int, layer_sizes: t.List[int] = DEFAULT_LAYER_SIZES, 
                 allow_negative_weights: bool = DEFAULT_ALLOW_NEGATIVE_WEIGHTS, 
                 max_weight: float = None, min_weight: float = None):
        super().__init__()
        
        self.input_shape: int = input_shape
        self.output_shape: int = output_shape
        self.layer_sizes: t.List[int] = layer_sizes
        
        self.allow_negative_weights: bool = allow_negative_weights

        if (max_weight is None and min_weight is None) or self.output_shape == 1:
            max_weight = Net.DEFAULT_MAX_WEIGHT
            min_weight = Net.DEFAULT_MIN_WEIGHT
        elif max_weight is None and min_weight is not None:
            max_weight = -min_weight * (self.output_shape - 1) + 1
        elif max_weight is not None and min_weight is None:
            min_weight = -(max_weight - 1) / (self.output_shape - 1)
        else:
            pass

        # Assert MAX > MIN
        assert max_weight > min_weight, f"'max_weight' must be larger than 'min_weight': provided {max_weight = } and {min_weight = }"
        # Assert MAX + (n - 1) * MIN
        assert max_weight + (self.output_shape - 1) * min_weight == 1, f"'max_weight' plus (n - 1) * 'min_weight' should be equal to 1: " +\
            f"{max_weight} + {self.output_shape - 1} * {min_weight} = {max_weight + (self.output_shape - 1) * min_weight}"
        
        self.allow_negative_weights: bool = min_weight < 0
        self.max_weight: float = max_weight
        self.min_weight: float = min_weight
        self._max_weight: float = max_weight - 1 / self.output_shape
        self._min_weight: float = min_weight - 1 / self.output_shape
        
        last_shape = self.input_shape
        for i, layer_size in enumerate(self.layer_sizes):
            setattr(self, f"fc{i}", nn.Linear(last_shape, layer_size))
            # setattr(self, f"dropout{i}", nn.Dropout(p=0.2))
            last_shape = layer_size
        self.fc_output = nn.Linear(last_shape, self.output_shape)
        # self.fc_output_1 = nn.Linear(last_shape, self.output_shape)
        # self.fc_output_2 = nn.Linear(2 * self.output_shape, self.output_shape)

    def forward(self, x):

        # Previous positions in assets (Skip Connection: FROM)
        x_pos = x[:, -self.output_shape:]

        for i, _ in enumerate(self.layer_sizes):
            x = getattr(self, f"fc{i}")(x)
            x = F.elu(x)
            
        x = self.fc_output(x)
        
        if self.output_shape == 1:
            if self.allow_negative_weights:
                # Boundaries: (-1, +1)
                output = F.tanh(x)
            else:
                # Boundaries: (0, +1)
                output = F.sigmoid(x)
        else:
            # Boundaries: (min_weight, max_weight), Sum: 1.0
            output = F.softmax(x, dim=1)
            output = (self._max_weight - self._min_weight) * output + self._min_weight + 1 / self.output_shape

        return output

    @staticmethod
    def translate(X: torch.Tensor, y: torch.Tensor, **kwargs) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return X, y

    @staticmethod
    def load(input_shape: int, output_shape: int, name: str = "latest"):
        
        PATH = Path(os.getcwd())
        model_dict = torch.load(PATH / f"outputs/models/{Net.MODEL_TYPE}-model-{name}-state.dict")
        
        net = Net(input_shape=input_shape, output_shape=output_shape)
        net.load_state_dict(model_dict)
        net.eval()
        
        return net

    def save(self, name: str) -> None:
        PATH = Path(os.getcwd())
        torch.save(self.state_dict(), PATH / f"outputs/models/{self.MODEL_TYPE}-model-{name}-state.dict")
        shutil.copy(PATH / f"outputs/models/{self.MODEL_TYPE}-model-{name}-state.dict", 
                    PATH / f"outputs/models/{self.MODEL_TYPE}-model-latest-state.dict")
        return


def example():

    # Define input and output sizes for neural network
    INPUT_SHAPE = 10
    OUTPUT_SHAPE = 2
    print(f"Input Shape = {INPUT_SHAPE}, Output Shape = {OUTPUT_SHAPE}")

    # Generate example data
    N_SAMPLES = 10
    X_sample = torch.randn(N_SAMPLES, INPUT_SHAPE)
    y_sample = torch.randn(N_SAMPLES, 1)
    print(f"{X_sample.shape = }, {y_sample.shape = }")
    
    X_sample_translated, y_sample_translated = Net.translate(X=X_sample, y=y_sample)
    print(f"{X_sample_translated.shape = }, {y_sample_translated.shape = }")

    X_sample_translated = X_sample_translated.to(device=DEVICE)
    y_sample_translated = y_sample_translated.to(device=DEVICE)

    # Initiate Network
    my_net = Net(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, layer_sizes=[5, 10, 5], allow_negative_weights=False).to(device=DEVICE)
    output = my_net.forward(x=X_sample_translated)
    print(f"{output.shape = }")
    print(f"{output = }")

    my_net = Net(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, layer_sizes=[8, 16, 8], allow_negative_weights=True).to(device=DEVICE)
    output = my_net.forward(x=X_sample_translated)
    print(f"{output.shape = }")
    print(f"{output = }")

    my_net = Net(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, layer_sizes=[8, 16, 8], 
                 allow_negative_weights=True, max_weight=2.0, min_weight=-1.0).to(device=DEVICE)
    output = my_net.forward(x=X_sample_translated)
    print(f"{output.shape = }")
    print(f"{output = }")
    
    return

if __name__ == "__main__":
    example()

