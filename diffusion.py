import torch as t
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from u_net import UNet, BasicUNet
from time_step_sampler import TimeStepSampler
from diffusion_transformer import DiffusionTransformer
from typing import Union
from functools import partial



def get_default_backbone(input_shape: tuple[int,...]):
    return DiffusionTransformer(input_shape=input_shape, num_heads=4, num_layers=6, d_model=128, d_ff=256, patch_size=4)

class Diffusion(t.nn.Module):

    def __init__(self, 
    input_shape: tuple[int,...],
    use_importance_sampling: bool=True, 
    training_time_steps: int=1000,
    backbone: Union[UNet, BasicUNet, DiffusionTransformer, None]=None):
        super().__init__()
        self.use_importance_sampling: bool = use_importance_sampling
        self.beta_start = 10 ** -4
        self.beta_stop = 10 ** -1
        self.training_time_steps = training_time_steps
        self.betas = t.linspace(self.beta_start, self.beta_stop, self.training_time_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = t.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = t.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = t.sqrt(1-self.alpha_bars)

        if backbone is None:
            self.backbone = get_default_backbone(input_shape)
        else:
            self.backbone = backbone

    def forward_process(self, x: t.Tensor, time_steps: t.Tensor) -> t.Tensor:
        """
        :param x: tensor with shape (batch, channels, d_image, d_image)
        :param time_steps: tensor with shape (batch,)
        """
        epsilon = t.normal(0, 1, x.shape)

        # Reshape alpha and beta terms to match image dimensions
        sqrt_alpha = self.sqrt_alpha_bars[time_steps].contiguous().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_bars[time_steps].contiguous().view(-1, 1, 1, 1)
        res = sqrt_alpha * x + sqrt_one_minus_alpha * epsilon  
        return res
    


    def forward(self, x, time_steps):
        return self.backbone(x, time_steps)
  

    def sample(self, sample_steps, batch_size: int=1, data_shape: tuple=(1, 28, 28)):
        """
        :param batch_size: int, number of samples to sample
        """
        self.eval()
        noised = t.randn(batch_size, *data_shape)
        with t.no_grad():
            for ts in range(sample_steps,0,-1):
                noised = self.forward(noised)
                # noised = sqrt_alpha * x + sqrt_one_minus_alpha * noise
                # noised - (sqrt_one_minus_alpha * noise)/sqrt_alpha = x
                # predicted_noise = predicted_image - noised
                # noised -= predicted_noise/self.sqrt_alpha_bars[ts-1]
        self.train()
        return t.clamp(noised, -1, 1)
  

def train(model,
data: Dataset, 
epochs: int=1, 
batch_size: int=64, 
print_intervals: int=1, 
debug: bool=False, 
batches: int=float('inf'), 
time_steps: int=None, 
lr: float=1e-4,
use_importance_sampling: bool=True):
    data = DataLoader(data, batch_size=batch_size, shuffle=True)
    
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    time_step_sampler = TimeStepSampler(model.training_time_steps, use_importance_sampling=use_importance_sampling)
    model.losses = []

    for epoch in range(epochs):
        for i ,(batch, labels) in enumerate(data):
            if i >= batches:
                break
            time_steps = time_step_sampler.sample_time_steps(batch.shape[0])
            noisy_data = model.forward_process(batch, time_steps)
            predicted_batch = model(noisy_data, time_steps)
            batch_losses = t.nn.MSELoss(reduction='none')(predicted_batch, batch).mean(dim=[1, 2, 3])
            # Ensure loss is a scalar
            loss = batch_losses.mean()
            
            if use_importance_sampling:
                time_step_sampler.update_losses(time_steps.detach().numpy(), batch_losses.detach().numpy())
            
            model.losses.append(loss.detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_intervals == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{batches}], Loss: {loss.item():.4f}")