import torch as t
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from image_net import SubNet

class Diffusion(t.nn.Module):

  def __init__(self, in_channels=1):
    super().__init__()
    self.beta_start = 10 ** -3
    self.beta_stop = 10 ** -1
    self.training_time_steps = 600
    self.betas = t.linspace(self.beta_start, self.beta_stop, self.training_time_steps)
    self.alphas = 1 - self.betas
    self.alpha_bars = t.cumprod(self.alphas, dim=0)
    self.sqrt_alpha_bars = t.sqrt(self.alpha_bars)
    self.sqrt_one_minus_alpha_bars = t.sqrt(1-self.alpha_bars)
    self.subnet = SubNet(in_channels=in_channels)

  def forward_process(self, x: t.Tensor, time_step: int) -> t.Tensor:
    """
    :param x: tensor with shape (batch, channels, d_image, d_image)
    """
    time_step_index = time_step - 1
    assert self.training_time_steps > time_step_index >= 0
    epsilon = t.normal(0,1,x.shape)
    return self.sqrt_alpha_bars[time_step_index] * x + (self.sqrt_one_minus_alpha_bars[time_step_index]) * epsilon

  def forward(self, x):
    return self.subnet.forward(x)

  def get_time_step(self,time_step_dim=1):
    return t.randint(1, self.training_time_steps+1, (time_step_dim,))
  
  def train(self,
  data: Dataset, 
  epochs: int=1, 
  batch_size: int =64, 
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
            noisy_data = self.forward_process(batch, random.randint(1,time_steps+1) if time_steps else self.get_time_step())
            noise = noisy_data - batch
            predicted_noise = self.forward(noisy_data)

            loss = t.nn.MSELoss()(noise, predicted_noise)
            self.losses.append(loss.detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if i % print_intervals == 0:
              print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{batches}], Loss: {loss.item():.4f}")
