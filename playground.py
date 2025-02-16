
from diffusion import Diffusion
import torch as t
import matplotlib.pyplot as plt
import torchvision
from time_step_sampler import TimeStepSampler
from training_scripts.train_mnist import nums_to_one_hot
LEARNING_RATE = 7e-5 
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
    print(f"num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.load_state_dict(t.load("weights/model_with_guidance.pth"))

    time_steps = t.tensor([100])
    for i, (x, y) in enumerate(data):
        x = model.forward_process(x,time_steps)
        if i > 7:
            break
        plt.imshow(x[0].squeeze().detach().numpy(), cmap='gray')
        plt.savefig(f"images/noised_num{y} iter{i}.png")
        plt.clf()
        x = model.forward(x, time_steps, guidance=nums_to_one_hot(t.Tensor([y]), 10))
        plt.imshow(x[0].squeeze().detach().numpy(), cmap='gray')
        plt.savefig(f"images/denoised_num{y} iter{i}.png")
        plt.clf()
