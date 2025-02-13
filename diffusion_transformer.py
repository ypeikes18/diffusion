import torch as t
import torch.nn as nn
from torch.nn import functional as F
from einops import einsum, rearrange
from torch.nn.functional import softmax
import math
from patchify import Patchify, Unpatchify
from time_step_embeddings import TimeStepEmbedder
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

class AdaLN(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.ln = nn.LayerNorm(d_model)
        self.scale = nn.Linear(d_model, d_model)
        self.shift = nn.Linear(d_model, d_model)

    def forward(self, x: t.Tensor, time_steps: t.Tensor) -> t.Tensor:
        """
        :param x: (batch_size, num_tokens, d_model)
        :param time_steps: (batch_size, 1)
        :return: (batch_size, num_tokens, d_model)
        """
        x = self.ln(x)
        x = self.scale(time_steps).unsqueeze(1) * x + self.shift(time_steps).unsqueeze(1)
        return x
    
class PositionalEncoding(nn.Module):
    pass


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k: int, d_model: int, num_heads: int, dropout_prob: float=0.1) -> None:
        super().__init__()
        self.d_k = d_k
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_prob)
        self.ln = nn.LayerNorm(d_model)
        self.attention_scale = 1 / math.sqrt(d_k)

        self.w_q = nn.Parameter(t.empty(num_heads, d_model, d_k))  
        self.w_k = nn.Parameter(t.empty(num_heads, d_model, d_k)) 
        self.w_v = nn.Parameter(t.empty(num_heads, d_model, d_k))
        self.w_o = nn.Parameter(t.empty(num_heads * d_k, d_model))
        
        self.b_q = nn.Parameter(t.zeros(num_heads, 1, d_k))
        self.b_k = nn.Parameter(t.zeros(num_heads, 1, d_k))
        self.b_v = nn.Parameter(t.zeros(num_heads, 1, d_k))
        self.b_o = nn.Parameter(t.zeros(d_model))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)
        nn.init.xavier_uniform_(self.w_o)
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        :param x: (batch_size, num_tokens, d_model)
        :return: (batch_size, num_tokens, d_model)
        """
        out = self.ln(x)
        q = einsum(out, self.w_q, "b tokens d_model, heads d_model d_k -> b heads tokens d_k")
        q = q + self.b_q
        k = einsum(out, self.w_k, "b tokens d_model, heads d_model d_k -> b heads tokens d_k")
        k = k + self.b_k

        attention_scores = einsum(q, k, "b heads q_tokens d_k, b heads k_tokens d_k -> b heads q_tokens k_tokens")
        attention_weights = t.softmax(attention_scores * self.attention_scale, dim=-1)

        v = einsum(out, self.w_v, "b tokens d_model, heads d_model d_k -> b heads tokens d_k")
        v = v + self.b_v

        z = einsum(attention_weights, v, "b heads q_tokens k_tokens, b heads k_tokens d_k -> b heads q_tokens d_k")

        # concat the output of the attention heads
        z = rearrange(z, "b heads tokens d_k -> b tokens (heads d_k)")
        z = einsum(z, self.w_o, "b tokens d_w, d_w d_model -> b tokens d_model")
        z = z + self.b_o
        return x + self.dropout(z)
    

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_prob: float=0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.in_layer = nn.Linear(d_model, d_ff)
        self.out_layer = nn.Linear(d_ff, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.ln(x)
        out = self.dropout(self.activation(self.in_layer(x)))
        out = self.dropout(self.out_layer(out))
        return x + out
        
    
class TransformerBlock(nn.Module):
    def __init__(self, *, d_model: int=64, num_heads: int=4, d_ff: int=128, d_k: int=16, dropout_prob: float=0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.multi_head_attention = MultiHeadAttention(d_k=d_k, d_model=d_model, num_heads=num_heads, dropout_prob=dropout_prob)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout_prob=dropout_prob)
        self.ln0 = AdaLN(d_model=d_model)
        self.ln1 = AdaLN(d_model=d_model)

    def forward(self, x: t.Tensor, time_steps: t.Tensor) -> t.Tensor:
        x = self.ln0(x, time_steps)
        x = self.multi_head_attention(x)
        x = self.ln1(x, time_steps)
        x = self.feed_forward(x)
        return x
    

class DiffusionTransformer(nn.Module):
    def __init__(self, *, input_shape: tuple, num_heads: int=4, num_layers: int=4, d_model: int=32, d_ff: int=128, patch_size: int=4, dropout_prob: float=0.1) -> None:
        super().__init__()
        self.patchify = Patchify(patch_size=patch_size, d_model=d_model, num_channels=input_shape[-3] if len(input_shape) >=3 else 1)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, d_k=d_model//num_heads, dropout_prob=dropout_prob) for _ in range(num_layers)])
        self.unpatchify = Unpatchify(patch_size=patch_size, d_model=d_model, output_shape=input_shape)
        self.layer_norm = nn.LayerNorm(d_model)
        self.norm = nn.Sigmoid()
        self.time_step_embedder = TimeStepEmbedder(d_embedding=d_model)

    def forward(self, x: t.Tensor, time_steps: t.Tensor) -> t.Tensor:
        x = self.patchify(x)
        # x = self.positional_encoding(x)
        time_steps = self.time_step_embedder(time_steps)
        for layer in self.transformer_blocks:
            x = layer(x, time_steps)
        x = self.layer_norm(x)
        x = self.unpatchify(x)
        x = self.norm(x)
        return x
