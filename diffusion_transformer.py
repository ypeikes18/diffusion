import torch as t
import torch.nn as nn
from torch.nn import functional as F
from einops import einsum, rearrange
from torch.nn.functional import softmax


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k: int, d_model: int, num_heads: int, dropout_prob: float=0.1) -> None:
        super().__init__()
        self.d_k = d_k
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_prob)
        self.attention_scale = 1 / t.sqrt(t.tensor(d_k, dtype=t.float32, requires_grad=False))
        self.w_q = nn.Parameter(t.empty(num_heads, d_model, d_k))  
        self.w_k = nn.Parameter(t.empty(num_heads, d_model, d_k)) 
        self.w_v = nn.Parameter(t.empty(num_heads, d_model, d_k))
        self.w_o = nn.Parameter(t.empty(num_heads * d_k, d_model))

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
        q = einsum(x, self.w_q, "b tokens d_model, heads d_model d_k -> b heads tokens d_k")
        k = einsum(x, self.w_k, "b tokens d_model, heads d_model d_k -> b heads tokens d_k")

        attention_scores = einsum(q, k, "b heads q_tokens d_k, b heads k_tokens d_k -> b heads q_tokens k_tokens")
        attention_weights = t.softmax(attention_scores * self.attention_scale, dim=-1)

        v = einsum(x, self.w_v, "b tokens d_model, heads d_model d_k -> b heads tokens d_k")
  
        z = einsum(attention_weights, v, "b heads q_tokens k_tokens, b heads k_tokens d_k -> b heads q_tokens d_k")
        
        # concat the output of the attention heads
        z = rearrange(z, "b heads tokens d_k -> b tokens (heads d_k)")
        z = einsum(z, self.w_o, "b tokens (heads d_k), (heads d_k) d_model -> b tokens d_model")
        return x + self.dropout(z)
    

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.in_layer = nn.Linear(d_model, d_ff)
        self.out_layer = nn.Linear(d_ff, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: t.Tensor) -> t.Tensor:
        out = nn.functional.gelu(self.in_layer(x))
        return x + self.ln(self.out_layer(out))
    

class Patchify(nn.Module):
    def __init__(self, *, patch_size: int=4, d_model: int=32 , num_channels: int=3) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size**2 * num_channels, d_model)


    def forward(self, x: t.Tensor) -> t.Tensor:
        # TODO add positional encoding to embeddings
        x = F.pad(x,(x.shape[-2]%self.patch_size, 0, x.shape[-1]%self.patch_size, 0))
        x = rearrange(
            x, 
            "b c (h ph) (w pw) -> b (h w) (ph pw c)",
            ph=self.patch_size,
            pw=self.patch_size
        )
        return self.projection(x)
    
class Unpatchify(nn.Module):
    def __init__(self, *, patch_size: int=4, d_model: int=32 , num_channels: int=3) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.projection = nn.Linear(d_model, patch_size**2 * num_channels)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.projection(x)
        x = rearrange(
            x, 
            "b (h w) (ph pw c) -> b c (h ph) (w pw)",
            ph=self.patch_size,
            pw=self.patch_size
        )
        return x
        
    
class TransformerBlock(nn.Module):
    def __init__(self, *, d_model: int=32, num_heads: int=4, d_ff: int=64, d_k: int=8) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.multi_head_attention = MultiHeadAttention(d_k=d_k, d_model=d_model, num_heads=num_heads)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)
        self.ln0 = nn.LayerNorm(d_model)
        self.ln1 = nn.LayerNorm(d_model)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.ln0(x)
        x = self.multi_head_attention(x)
        x = self.ln1(x)
        x = self.feed_forward(x)
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, *, num_channels: int=3, num_heads: int=4, num_layers: int=4, d_model: int=32, d_ff: int=128, patch_size: int=4) -> None:
        super().__init__()
        self.patchify = Patchify(patch_size=patch_size, d_model=d_model, num_channels=num_channels)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, d_k=d_model//num_heads) for _ in range(num_layers)])
        self.unpatchify = Unpatchify(patch_size=patch_size, d_model=d_model, num_channels=num_channels)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.patchify(x)
        for layer in self.transformer_blocks:
            x = layer(x)
        x = self.unpatchify(x)
        return x

    
    


    