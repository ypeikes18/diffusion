from einops import rearrange
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math

class Patchify(nn.Module):
    def __init__(self, *, patch_size: int=4, d_model: int=32 , num_channels: int=3) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size**2 * num_channels, d_model)


    def forward(self, x: t.Tensor) -> t.Tensor:
        # TODO add positional encoding to embeddings
        image_height = x.shape[-2]
        image_width = x.shape[-1]
        x = F.pad(x,(image_width%self.patch_size, 0, image_height%self.patch_size, 0))
        x = rearrange(
            x, 
            "b c (num_patches_h ph) (num_patches_w pw) -> b (num_patches_h num_patches_w) (ph pw c)",
            num_patches_h = int(x.shape[-2] / self.patch_size),
            num_patches_w = int(x.shape[-1] / self.patch_size),
            ph=self.patch_size,
            pw=self.patch_size
        )
        out = self.projection(x)
        return out


class Unpatchify(nn.Module):
    def __init__(self, *, patch_size: int=8, d_model: int=64 , output_shape: tuple) -> None:
        super().__init__()
        self.output_shape = output_shape
        self.patch_size = patch_size
        self.num_patches = math.ceil(self.output_shape[-2]/self.patch_size)*math.ceil(self.output_shape[-1]/self.patch_size)
        self.projection = nn.Linear(
            d_model, 
            int(t.prod(t.tensor(self.output_shape))//self.num_patches)
        )
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.projection(x)
        # self.num_patches, (c h w)/num_patches = c h w
        # c h w/num_patches 
        x = rearrange(
            x, 
            "b patches d_patch-> b (patches d_patch)"
        )
        x = rearrange(
            x, 
            "b (c h w) -> b c h w", 
            c=self.output_shape[-3] if len(self.output_shape) >= 3 else 1, 
            h=self.output_shape[-2], 
            w=self.output_shape[-1])
        return x