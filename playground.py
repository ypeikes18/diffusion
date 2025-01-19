
# %%
from diffusion import Diffusion
from utils.debug_utils import get_moving_average
from mnist import MNISTDataset
import torch as t
import matplotlib.pyplot as plt
import statistics

# %%
model = Diffusion(in_channels=1, use_importance_sampling=True)
data = MNISTDataset()
model.train(data, epochs=1, batch_size=64, print_intervals=1, debug=True, lr=3e-5, batches=200)


# x = t.randn(1, 1, 28, 28)
# def sample(x):
#     for i in range(1000): 
#       x  += model.forward(x)
#     return x
# s = sample(x).squeeze().squeeze().squeeze().detach().numpy()

# plt.imshow(s*255, cmap='gray')
# plt.savefig("sample.png")
# plt.imshow(x*255, cmap='gray')
# plt.savefig("x.png")
# print(s)
model.losses = get_moving_average(model.losses, 10)

# %%
plt.clf()
plt.plot(model.losses)
plt.savefig("loss_plot_no_is.png")
# %%
