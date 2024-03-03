from loguru import logger
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import time
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import typing as t

from . import utils


def get_device() -> torch.device:

    if torch.cuda.is_available():
        logger.info("Running on the GPU")
        n_gpu = 0 # different if you have more than 1
        assert n_gpu <= (torch.cuda.device_count() - 1)
        logger.info(f"Using gpu={n_gpu} out of {torch.cuda.device_count()}")
        device = torch.device(f"cuda:{n_gpu}")
    else:
        logger.info("Running on the CPU")
        device = torch.device("cpu")
    
    return device

BATCH_SIZE: int = 64
DEVICE: torch.device = get_device()
EARLY_STOPPING_PATIENCE: int = 10
EARLY_STOPPING_MIN_DELTA: float = 0.0
EARLY_STOPPING_MIN_PERIODS: int = 150 # 0
EPOCHS: int = 2_000
LEARNING_RATE: float = 0.001
PATH: Path = utils.get_path(path=__file__)
SHUFFLE: bool = False
TRAIN_SIZE: float = 0.7


def convert_data_to_tensors(X: pd.DataFrame, y: pd.DataFrame) -> t.Tuple[torch.Tensor, torch.Tensor]:    
    X_tensor = torch.Tensor(X.values) if isinstance(X, pd.DataFrame) else X
    y_tensor = torch.Tensor(y.values) if isinstance(y, pd.DataFrame) else y
    return X_tensor, y_tensor

def split_data(X: torch.Tensor | pd.DataFrame, y: torch.Tensor | pd.DataFrame, train_size: float = TRAIN_SIZE, shuffle: bool = SHUFFLE)\
    -> t.Tuple[torch.Tensor | pd.DataFrame, torch.Tensor | pd.DataFrame, torch.Tensor | pd.DataFrame, torch.Tensor | pd.DataFrame]:
    return train_test_split(X, y, train_size=train_size, shuffle=shuffle)


class EarlyStopping:
    
    DEFAULT_PATIENCE: int = 5
    DEFAULT_MIN_DELTA: int = 0
    
    def __init__(self, patience: int = DEFAULT_PATIENCE, min_delta: int = DEFAULT_MIN_DELTA, 
                 maximize: bool = False, min_periods: int = 0):

        self.patience: int = patience
        self.min_delta: int = min_delta
        self.maximize: bool = maximize
        self.min_periods: int = min_periods

        # Benchmark Loss
        self.best_loss: float = -np.inf if maximize else np.inf

        # Number of Updates - used for min periods before applying early stopping
        self.updates: int = 0
        # Counter to count updates since last minimum
        self.counter: int = 0
        # Boolean to indicate whether to stop or continue training
        self.early_stop: bool = False

    def __call__(self, loss: float) -> None:

        """ Updates state and sets `early_stop` to True when the loss has not improved by `min_loss` in `patience` updates. """

        self.updates += 1

        if (((loss + self.min_delta) < self.best_loss) and not self.maximize) or (((loss - self.min_delta) > self.best_loss) and self.maximize):
            self.best_loss = loss
            self.counter = 0
            return

        if (((loss + self.min_delta) > self.best_loss) and not self.maximize) or (((loss - self.min_delta) < self.best_loss) and self.maximize):
            self.counter += 1
            if (self.counter >= self.patience) and (self.updates > self.min_periods):
                self.early_stop = True
        
        return


def get_prediction(net: nn.Module, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    X = X.to(device=DEVICE)
    y = y.to(device=DEVICE)

    w = net(X)
    return w


def fwd_pass(net: nn.Module, loss_fn, X: torch.Tensor, y: torch.Tensor, optimizer: torch.optim = None, do_train: bool = False, **kwargs):
    
    X = X.to(device=DEVICE)
    y = y.to(device=DEVICE)
    
    if do_train:
        net.zero_grad()

    w = net(X)
    r = torch.sum(w * y, dim=1)

    active = torch.any((w > 1e-5)[:, :-1], dim=1)
    matches = [r_ >= 0 for r_, a in zip(r, active) if a]
    hit_rate = matches.count(True) / (len(matches) + 1e-10)
    if len(matches) == 0:
        print(f"{w = }")
        print(f"{r = }")
        print(f"{matches = }")
    
    loss = loss_fn(weights=w, returns=y, **kwargs)

    if do_train:
        assert optimizer is not None, f"Optimizer not provided in fwd_pass with training"
        loss.backward()
        optimizer.step()
    
    return loss, hit_rate


def train(net: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor, 
          loss_fn, store: bool = False, lr: bool = LEARNING_RATE, batch_size: int = BATCH_SIZE, epochs: int = EPOCHS,
          maximize_loss: bool = False, verbose: bool = True, **loss_kwargs):

    optimizer = optim.Adam(net.parameters(), lr=lr, maximize=maximize_loss)
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE, 
        min_delta=EARLY_STOPPING_MIN_DELTA, 
        maximize=maximize_loss,
        min_periods=EARLY_STOPPING_MIN_PERIODS,
    )
    
    MODEL_NAME: str = f"{int(time.time())}"

    if verbose:
        logger.info(f"Training: {MODEL_NAME!r}")

    with open(PATH / f"scripts/outputs/models/{net.MODEL_TYPE}-model-{MODEL_NAME}.log", "a") as log:
        
        for epoch in range(epochs):

            net.train()
            for i in tqdm(range(0, len(X_train), batch_size)):

                batch_X = X_train[i:(i + batch_size)].to(device=DEVICE)
                batch_y = y_train[i:(i + batch_size)].to(device=DEVICE)

                acc, loss = fwd_pass(net=net, loss_fn=loss_fn, X=batch_X, y=batch_y, optimizer=optimizer, do_train=True, **loss_kwargs)

            net.eval()
            loss, hit_rate = evaluate(net=net, loss_fn=loss_fn, X=X_train, y=y_train)
            val_loss, val_hit_rate = evaluate(net=net, loss_fn=loss_fn, X=X_test, y=y_test)
            if store:
                log.write(f"{MODEL_NAME},{time.time():.3f},{epoch},{loss:.6f},{hit_rate:.6f},{val_loss:.6f},{val_hit_rate:.6f}\n")
            if verbose:
                logger.info(f"Epoch: {epoch} / {epochs}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}," +\
                            f"Hit Rate: {hit_rate:.2%}, Val Hit Rate: {val_hit_rate:.2%}")

            # early stopping
            early_stopping(loss)
            if early_stopping.early_stop:
                if verbose:
                    logger.info(f"Early Stopping reached @ {epoch = }! Best Loss: {early_stopping.best_loss}, Loss: {loss}," +\
                                f"Val Loss: {val_loss}, Hit Rate: {hit_rate:.2%}, Val Accuracy: {val_hit_rate:.2%}")
                break

    if store:
        net.save(name=MODEL_NAME)
        shutil.copy(PATH / f"scripts/outputs/models/{net.MODEL_TYPE}-model-{MODEL_NAME}.log", 
                    PATH / f"scripts/outputs/models/{net.MODEL_TYPE}-model-latest.log")
    else:
        os.remove(PATH / f"scripts/outputs/models/{net.MODEL_TYPE}-model-{MODEL_NAME}.log")

    return
    

def test(net: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, loss_fn, size: int = None) -> t.Tuple[float, torch.Tensor]:

    if size is None:
        X, y = X_test, y_test
    else:
        # Get random split of test data of size `size`
        random_start = np.random.randint(X_test.shape[0] - size)
        X, y = X_test[random_start:(random_start + size)], y_test[random_start:(random_start + size)]
    
    return evaluate(net=net, loss_fn=loss_fn, X=X, y=y)


def evaluate(net: nn.Module, loss_fn, X: torch.Tensor, y: torch.Tensor, **kwargs) -> t.Tuple[float, torch.Tensor]:

    with torch.no_grad():
        loss, hit_rate = fwd_pass(net=net, loss_fn=loss_fn, X=X, y=y, do_train=False, **kwargs)

    return loss, hit_rate


def create_sequence(
    X: torch.Tensor, y: torch.Tensor, sequence_length: int, step_size: int = 1,
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    
    num_samples: int = X.shape[0]
    num_variables: int = X.shape[1]
    
    # Create a new dataset with overlapping sequences
    new_X = []
    new_y = []
    for i in range(sequence_length, num_samples, step_size):
        # Fetch last `sequence_length` of X
        new_X.append(X[(i - sequence_length):i])
        # Fetch i-th value of y
        new_y.append(y[i])

    # Convert lists to tensors
    new_X = torch.stack(new_X)
    new_y = torch.stack(new_y)
        
    return new_X, new_y
