import torch as t
import torch.nn as nn

# Based on  the embeddings desribed in "Attention is all you need"
# https://arxiv.org/pdf/1706.03762
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        # shape (num_tokens,1)
        positions = t.arange(0,x.shape[-2], requires_grad=False).unsqueeze(-1)
        # shape (1, d_model)
        denominator = 10000 ** (t.arange(0, self.d_model, 1, requires_grad=False) / self.d_model).unsqueeze(-2)
        # shape (num_tokens, d_model)
        func_args = positions / denominator

        pe = t.zeros_like(func_args, requires_grad=False)
        pe[:, ::2] += t.sin(func_args[:,::2])
        pe[:, ::2] += t.cos(func_args[:,1::2])
        
        return x + pe
