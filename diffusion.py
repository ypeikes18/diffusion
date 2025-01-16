import torch as t
import random
import matplotlib.pyplot as plt
import numpy as np
from mnist import get_data_loader

import torch.nn as nn
from image_net import SubNet
from utils.debug_utils import debug_print

class Diffusion(t.nn.Module, in_channels=1):

  def __init__(self):
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
    assert self.training_time_steps >= time_step > 0
    epsilon = t.normal(0,1,x.shape)
    return self.sqrt_alpha_bars[time_step_index] * x + (self.sqrt_one_minus_alpha_bars[time_step_index]) * epsilon

  def forward(self, x):
    return self.subnet.forward(x)

  def get_time_step(self):
    return np.random.randint(1, self.training_time_steps+1)


model = Diffusion(in_channels=1)
optimizer = t.optim.Adam(model.parameters(), lr=1e-4)
debug=False
epochs = 1
batches = 1000
batch_size = 32
print_intervals = 1



losses = []
for epoch in range(epochs):
    for i ,(batch, labels) in enumerate(get_data_loader()):
        # For now hard code the time step to 1 to see if it works
        noisy_data = model.forward_process(batch, 1)
        noise = noisy_data - batch
        predicted_noise = model.forward(noisy_data)

        loss = t.nn.MSELoss()(noise, predicted_noise)
        losses.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i % print_intervals == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{batches}], Loss: {loss.item():.4f}")

breakpoint()