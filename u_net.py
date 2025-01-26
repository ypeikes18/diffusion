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
    kwargs = asdict(ConvNdKwargs(**kwargs))
    # TODO: the above might not handle defaults
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
        self.relu = nn.ReLU()


    def forward(self, x):
        return self.relu(self.conv(x))
    
    
class UpBlock(nn.Module):

    def __init__(self, *,
    in_channels: int=3, 
    out_channels: int=3, 
    conv_dims: Literal[1,2,3]=2, 
    conv_dims_out_shape: tuple[int], 
    use_conv: bool = True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_dims_out_shape = conv_dims_out_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        if use_conv:
            self.conv = create_nd_conv(conv_dims=conv_dims, in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
        self.use_conv = use_conv
        

    def forward(self, x, skip_connection):
        breakpoint()
        x = F.interpolate(x, size=self.conv_dims_out_shape, mode='nearest')
        x = t.cat([x, skip_connection], dim=1)
        if self.use_conv:
            x = self.conv(x)
        return self.relu(x)
    

class UNet(nn.Module):
    def __init__(self, *, in_channels, conv_dims_out_shape: tuple[int], num_up_down_blocks=3, conv_dims: Literal[1,2,3]=2, channel_multiplier: int=32):
        super().__init__()
        self.in_channels = in_channels
        self.conv_dims_out_shape = conv_dims_out_shape
        self.num_up_down_blocks = num_up_down_blocks
        self.conv_dims = conv_dims
        self.channel_multiplier = channel_multiplier
        self.down_blocks = nn.ModuleDict()
        self.up_blocks = nn.ModuleDict()
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
        # (64, 1, 28, 28) -> (64, 64, 14, 14) -> (64, 128, 7, 7) -> (64, 256, 4, 4)
        # -> (64, 128, 7, 7) -> (64, 256, 14, 14) -> (64, 1, 28, 28)
        for i in range(self.num_up_down_blocks):
            skip_connections.append(x.clone())
            print(f"down{i} shape: {x.shape}")
            x = self.down_blocks[f'down{i}'](x)
        for i in range(self.num_up_down_blocks):
            x = self.up_blocks[f'up{i}'](x, skip_connections.pop())
            print(f"up{i} shape: {x.shape}")
        return x
    

    def _initialize_blocks(self) -> None:
        in_channels: int = self.in_channels            
        
        for i in range(self.num_up_down_blocks):
            channel_multiplier_factor: int = self.channel_multiplier * (2 ** i)
            down_block_out_channels: int = self.in_channels * channel_multiplier_factor
            conv_dims_out_shape: tuple[int] = tuple([math.ceil(d/2**i) for d in self.conv_dims_out_shape[-self.conv_dims:]]) 
            
            self.down_blocks[f'down{i}'] = DownBlock(
                in_channels=in_channels,
                out_channels=down_block_out_channels,
                conv_dims=self.conv_dims
            )
            
            self.up_blocks[f'up{(self.num_up_down_blocks-i)-1}'] = UpBlock(
                # TODO explain the 1.5
                in_channels=int(down_block_out_channels * 1.5),
                out_channels=in_channels,
                conv_dims=self.conv_dims,
                conv_dims_out_shape=conv_dims_out_shape,
                use_conv=True
            )
            in_channels = down_block_out_channels
