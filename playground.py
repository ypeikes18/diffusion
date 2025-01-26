
# %%
from diffusion import Diffusion, train
from utils.debug_utils import get_moving_average
from mnist import MNISTDataset
import torch as t
import matplotlib.pyplot as plt

LEARNING_RATE = 2e-5
NOISE_STEPS = 10

if __name__ == "__main__":

    data = MNISTDataset()
    model = Diffusion(in_channels=1, use_importance_sampling=True, training_time_steps=500, num_up_down_blocks=3)
    train(model,data, epochs=1, batch_size=64, print_intervals=1, debug=True, lr=LEARNING_RATE)


    img = data.train_images[5]
    img = model.forward_process(img, NOISE_STEPS)
    plt.imshow(img.squeeze().detach().numpy(), cmap='gray')
    plt.savefig("images/img.png")
    plt.clf()   

    denoised = model.forward(img)
    plt.imshow(denoised.squeeze().detach().numpy(), cmap='gray')
    plt.savefig("images/denoised.png")
    plt.clf()   
