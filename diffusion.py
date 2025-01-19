import torch as t
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from u_net import UNet
import matplotlib.pyplot as plt
from time_step_sampling import TimeStepSampler



class Diffusion(t.nn.Module):

    def __init__(self, in_channels: int=1, use_importance_sampling: bool=True, training_time_steps: int=1000):
        super().__init__()
        self.use_importance_sampling: bool = use_importance_sampling
        self.beta_start = 10 ** -3
        self.beta_stop = 10 ** -1
        self.training_time_steps = 1000
        self.betas = t.linspace(self.beta_start, self.beta_stop, self.training_time_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = t.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = t.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = t.sqrt(1-self.alpha_bars)
        self.subnet = UNet(in_channels=in_channels)
        self.time_step_sampler = TimeStepSampler(self.training_time_steps, use_importance_sampling=use_importance_sampling)


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


    def forward(self, x):
        return self.subnet.forward(x)

    
    def train(self,
    data: Dataset, 
    epochs: int=1, 
    batch_size: int=64, 
    print_intervals: int=1, 
    debug: bool=False, 
    batches: int=float('inf'), 
    time_steps: int=None, 
    lr: float=1e-4):
        data = DataLoader(data, batch_size=batch_size, shuffle=True)
        optimizer = t.optim.Adam(self.parameters(), lr=lr)
        self.losses = []

        for epoch in range(epochs):
            for i ,(batch, labels) in enumerate(data):
                if i >= batches:
                    break
                time_steps = self.time_step_sampler.sample_time_steps(batch.shape[0])
                noisy_data = self.forward_process(batch, time_steps)

                
                noise = noisy_data - batch
                predicted_noise = self.forward(noisy_data)

                batch_losses = t.nn.MSELoss(reduction='none')(noise, predicted_noise).mean(dim=[1, 2, 3])
                # Ensure loss is a scalar
                loss = batch_losses.mean()
              
                if self.use_importance_sampling:
                    self.time_step_sampler.update_losses(time_steps.detach().numpy(), batch_losses.detach().numpy())
                
                self.losses.append(loss.detach().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if i % print_intervals == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{batches}], Loss: {loss.item():.4f}")
  
  
    def sample(self, batch_size: int=1, data_shape: tuple=(1, 28, 28)):
        """
        :param time_steps: int, number of time steps to sample
        :param batch_size: int, number of samples to sample
        """
        x = t.randn(batch_size, *data_shape)
        for i in range(self.training_time_steps):
            x += self.forward(x)
        return x
  