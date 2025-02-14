
from diffusion import Diffusion, train
import torch as t
import matplotlib.pyplot as plt
import torchvision
from time_step_sampler import TimeStepSampler

LEARNING_RATE = 5e-5 
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
    ts = TimeStepSampler(model.training_time_steps, use_importance_sampling=True)
    model.load_state_dict(t.load("weights/model.pth"))
    print(f"num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # train(model,data, epochs=2, batch_size=64, print_intervals=1, debug=True, lr=LEARNING_RATE)
    # t.save(model.state_dict(), "weights/model.pth")
    x, y = next(iter(data))

    time_steps = t.tensor([50])

    x = model.forward_process(x,time_steps)
    plt.imshow(x[0].squeeze().detach().numpy(), cmap='gray')
    plt.savefig("images/noised.png")
    plt.clf()
    x = model.forward(x, time_steps)
    plt.imshow(x[0].squeeze().detach().numpy(), cmap='gray')
    plt.savefig("images/denoised.png")
    plt.clf()
