
# %%
from diffusion import Diffusion
from mnist import MNISTDataset
import torch as t
import matplotlib.pyplot as plt

# %%
data = MNISTDataset()
plt.clf()
model = Diffusion(in_channels=1, use_importance_sampling=True, training_time_steps=1000)
model.train(data, epochs=1, batch_size=64, print_intervals=1, debug=True, lr=1e-4, batches=200)
t.save(model.state_dict(), 'model_weights.pth')
model.load_state_dict(t.load('model_weights.pth'))
sample = model.sample(1000)
plt.imshow(sample.squeeze().detach().numpy(), cmap='gray')
plt.savefig("images/sample.png")
print(sample)


