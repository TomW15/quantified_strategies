import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import typing as t


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
            # x = F.relu(x)
            # x = F.leaky_relu(x)
            x = F.elu(x)
            # x = getattr(self, f"dropout{i}")(x)
        
        # x = self.fc_output_1(x)
        # x = F.elu(x)

        # Skip Connection: TO
        # x = x + x_pos
        # print(f"{x.shape = }")
        # print(f"{x_pos.shape = }")
        # x = torch.concat((x, x_pos), dim=1)
        
        # x = self.fc_output_2(x)

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
        
        PATH = Path(__file__).parent.parent
        model_dict = torch.load(PATH / f"scripts/outputs/models/{Net.MODEL_TYPE}-model-{name}-state.dict")
        
        net = Net(input_shape=input_shape, output_shape=output_shape)
        net.load_state_dict(model_dict)
        net.eval()
        
        return net

    def save(self, name: str) -> None:
        PATH = Path(__file__).parent.parent
        torch.save(self.state_dict(), PATH / f"scripts/outputs/models/{self.MODEL_TYPE}-model-{name}-state.dict")
        shutil.copy(PATH / f"scripts/outputs/models/{self.MODEL_TYPE}-model-{name}-state.dict", 
                    PATH / f"scripts/outputs/models/{self.MODEL_TYPE}-model-latest-state.dict")
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
