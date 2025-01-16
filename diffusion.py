import torch as t
import random
import matplotlib.pyplot as plt
import numpy as np
from mnist import get_data_loader

import torch.nn as nn
from image_net import SubNet

class Diffusion(t.nn.Module):

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
    self.subnet = SubNet()

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


def add_channel_with_value(tensor, value):
  """Adds a channel filled with a specific value to a tensor.

  Args:
    tensor: The input tensor with shape (batch, channels, rows, columns).
    value: The value to fill the new channel with.

  Returns:
    The tensor with the added channel.
  """
  batch_size, channels, rows, cols = tensor.shape

  new_channel = t.full((batch_size, 1, rows, cols), value, dtype=tensor.dtype, device=tensor.device)
  result = t.cat([tensor, new_channel], dim=1)
  return result



model = Diffusion()
optimizer = t.optim.Adam(model.parameters(), lr=1e-4)
debug=False
epochs = 1
batches = 1000
batch_size = 32
print_intervals = 1


def debug_print(string):
  if debug:
    print(string)
losses = []
for epoch in range(epochs):
    for i ,(batch, labels) in enumerate(get_data_loader()):
        time_step  = random.sample([1], 1)[0]
        debug_print(f"BATCH SHAPE {batch.shape}")
        debug_print(f"TIME_STEP {time_step}")
        batch = add_channel_with_value(batch, time_step)
        debug_print(f"BATCH SHAPE {batch.shape}")
        noisy_data = model.forward_process(batch, model.get_time_step())
        debug_print(f"NOISY_DATA SHAPE {batch.shape}")
        noise = noisy_data - batch
        predicted_noise = model.forward(noisy_data)

        loss = t.nn.MSELoss()(noise, predicted_noise)
        losses.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i % print_intervals == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{batches}], Loss: {loss.item():.4f}")

plt.plot(losses, label="loss")
plt.show()
