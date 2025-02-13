import torch as t
import torch.nn as nn
from functools import lru_cache
import math
from einops import rearrange
# based largely on https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L27

class TimeStepEmbedder(nn.Module):
    def __init__(self, d_embedding: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.d_embedding = d_embedding
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, d_embedding),
            nn.SiLU(),
            nn.Linear(d_embedding, d_embedding),
        )
        

    def forward(self, timesteps: t.Tensor) -> t.Tensor:
        return self.mlp(self.get_timestep_embedding(timesteps))
        

    def get_timestep_embedding(self, timesteps: t.Tensor) -> t.Tensor:
        half_dims = self.frequency_embedding_size // 2
        frequency_embeddings = t.exp(
            t.arange(start=0, end=half_dims) * 
            -math.log(10000) / half_dims
        )
        timesteps = rearrange(timesteps, 'b -> b 1')
        func_args = timesteps * frequency_embeddings
        embeddings = t.cat([t.sin(func_args), t.cos(func_args)], dim=-1)
        if half_dims % 2 == 1:
            embeddings = t.cat([embeddings, t.sin(func_args)], dim=-1)
        return embeddings
