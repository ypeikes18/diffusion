
# %%
from diffusion import Diffusion
from utils.debug_utils import get_moving_average

# %%
model = Diffusion(in_channels=1)
model.train(epochs=1, batch_size=64, print_intervals=1, debug=True, batches=100, time_steps=600)
model.losses = get_moving_average(model.losses, 10)

# %%
import matplotlib.pyplot as plt
plt.plot(model.losses)
plt.savefig("loss_plot.png")
# %%
import torch
print(f"ROCm available: {hasattr(torch, 'hip')}")
# %%
