
# %%
from diffusion import Diffusion
from utils.debug_utils import get_moving_average
from mnist import MNISTDataset
# %%
model = Diffusion(in_channels=1)
ts = model.get_time_step(10)
print(ts)
# %%
data = MNISTDataset()
print(data)
print(data.train_images.shape)
model = Diffusion(in_channels=1)

model.train(data, epochs=1, batch_size=64, print_intervals=1, debug=True, lr=3e-5)
model.losses = get_moving_average(model.losses, 10)

# %%
import matplotlib.pyplot as plt
plt.plot(model.losses)
plt.savefig("loss_plot.png")
# %%
import statistics

means = [statistics.mean(model.losses[i:i+500]) for i in range(0, len(model.losses), 500)]
print(len(means))

plt.plot(means)
print([int(i) for i in means])
plt.savefig("loss_plot.png")

# %%
