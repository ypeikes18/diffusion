import torch as t
import torch.nn as nn
from typing import Literal
from dataclasses import dataclass, asdict
from torch.functional import F
import math

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


class DownBlock(nn.Module):

    def __init__(self, *, 
    in_channels: int=3, 
    out_channels: int=3, 
    conv_dims: Literal[1,2,3]=2):
        super().__init__()
        self.conv = create_nd_conv(conv_dims=conv_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.activation = nn.SiLU()


    def forward(self, x):
        return self.activation(self.conv(x))
    
    
class UpBlock(nn.Module):

    def __init__(self, *,
    in_channels: int=3, 
    out_channels: int=3, 
    conv_dims: Literal[1,2,3]=2, 
    conv_dims_out_shape: tuple[int]):
        super().__init__()
        self.activation = nn.SiLU()
        self.conv_dims_out_shape = conv_dims_out_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = create_nd_conv(conv_dims=conv_dims, in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x, skip_connection):
        x = F.interpolate(x, size=self.conv_dims_out_shape, mode='nearest')
        x = self.conv(x)
        x += skip_connection
        return self.activation(x)
    

class UNet(nn.Module):
    def __init__(self, *, in_channels, conv_dims_out_shape: tuple[int], num_up_down_blocks=3, conv_dims: Literal[1,2,3]=2, channel_multiplier: int=32):
        super().__init__()
        self.in_channels = in_channels
        self.conv_dims_out_shape = conv_dims_out_shape
        self.num_up_down_blocks = num_up_down_blocks
        self.conv_dims = conv_dims
        self.channel_multiplier = channel_multiplier
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.out_layer = nn.Sequential(
            create_nd_conv(conv_dims=conv_dims, in_channels=self.in_channels, out_channels=channel_multiplier, kernel_size=3, stride=1, padding=1), 
            nn.SiLU(),
            create_nd_conv(conv_dims=conv_dims, in_channels=channel_multiplier, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1), 
        )
        self._initialize_blocks()
        self._validate_initialization()

        
    def _validate_initialization(self) -> None:
        assert self.conv_dims in [1,2,3]
        assert len(self.conv_dims_out_shape) == self.conv_dims


    def forward(self, x):
        """
        :param x: input tensor of shape (batch_size, channels, *conv_dims_out_shape)
        """
        skip_connections = []
        for down_block in self.down_blocks:
            skip_connections.append(x.clone())
            x = down_block(x)
        for i in range(self.num_up_down_blocks-1,-1,-1):
            x = self.up_blocks[i](x, skip_connections.pop())
        return self.out_layer(x)
    

    def _initialize_blocks(self) -> None:
        in_channels: int = self.in_channels            
        
        for i in range(self.num_up_down_blocks):
            channel_multiplier_factor: int = self.channel_multiplier * (2 ** i)
            down_block_out_channels: int = self.in_channels * channel_multiplier_factor
            conv_dims_out_shape: tuple[int] = tuple([math.ceil(d/2**i) for d in self.conv_dims_out_shape[-self.conv_dims:]]) 
            self.down_blocks.append(DownBlock(
                in_channels=in_channels,
                out_channels=down_block_out_channels,
                conv_dims=self.conv_dims
            ))
            
            self.up_blocks.append(UpBlock(
                in_channels=down_block_out_channels,
                out_channels=in_channels,
                conv_dims=self.conv_dims,
                conv_dims_out_shape=conv_dims_out_shape
            ))
            in_channels = down_block_out_channels
