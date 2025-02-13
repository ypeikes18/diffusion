
from diffusion import Diffusion, train
from utils.debug_utils import get_moving_average
# from mnist import MNISTDataset
import torch as t
import matplotlib.pyplot as plt
import torchvision

LEARNING_RATE = 1e-4
NOISE_STEPS = 10

if __name__ == "__main__":

    data = torchvision.datasets.MNIST(
        root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor()
    )

    model = Diffusion(
        input_shape=(1, 28, 28), 
        use_importance_sampling=True, 
        training_time_steps=500, 
    )

    print(f"num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    train(model,data, epochs=1, batch_size=64, print_intervals=1, debug=True, lr=LEARNING_RATE, batches=500)
    # t.save(model.state_dict(), "./model.pth")
    x, y = next(iter(data))
    x = model.forward_process(x,t.tensor([15]))
    plt.imshow(x[0].squeeze().detach().numpy(), cmap='gray')
    plt.savefig("images/noised.png")
    plt.clf()
    x = model.forward(x, model.get_time_steps(t.tensor([15])))
    plt.imshow(x[0].squeeze().detach().numpy(), cmap='gray')
    plt.savefig("images/denoised.png")
    plt.clf()
