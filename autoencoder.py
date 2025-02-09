import torch as t
from torch.utils.data import DataLoader
from torch import nn
from dataclasses import dataclass, asdict
from typing import Literal

@dataclass
class ConvNdKwargs:
    in_channels: int=3
    out_channels: int=3
    kernel_size: int=3
    stride: int=1
    padding: int=1

def create_nd_conv(conv_dims: Literal[1,2,3]=2, **kwargs: ConvNdKwargs) -> nn.Module:
    # populates kwargs with defaults for any ommited parameters
    kwargs = asdict(ConvNdKwargs(**kwargs))
    if conv_dims == 1:
        return nn.Conv1d(**kwargs)
    elif conv_dims == 2:
        return nn.Conv2d(**kwargs)
    elif conv_dims == 3:
        return nn.Conv3d(**kwargs)
    else:
        raise ValueError(f"Unsupported number of dimensions: {conv_dims}")

class Autoencoder(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = t.nn.Sequential(
            t.nn.Linear(28*28, 128),
            t.nn.ReLU(),
            t.nn.Linear(128, 64),
            t.nn.ReLU(),
            t.nn.Linear(64, 32),
            t.nn.ReLU(),
        )

class Encoder(t.nn.Module):
    def __init__(self, conv_dims: int, in_channels: int, out_channels: int):
        super().__init__()
        self.encoder = t.nn.Sequential(
            create_nd_conv(conv_dims, in_channels, in_channels*2),
            nn.ReLU(),
            create_nd_conv(conv_dims, in_channels*2, in_channels*4),
            nn.ReLU(),
            create_nd_conv(conv_dims, in_channels*4, out*8),
            nn.ReLU(),
        )


def train(model, data, epochs=1, batch_size=32):
    data = DataLoader(data, batch_size=batch_size, shuffle=True)
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for i ,(batch, labels) in enumerate(data):
            reconstructed = model(batch)
            optimizer.zero_grad()
            loss = t.nn.MSELoss(reconstructed, batch)
            loss.backward()
            optimizer.step()


