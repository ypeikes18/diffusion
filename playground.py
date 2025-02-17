
from diffusion import Diffusion
import torch as t
import matplotlib.pyplot as plt
import torchvision
from time_step_sampler import TimeStepSampler
from training_scripts.train_mnist import nums_to_one_hot
from training_scripts.train_mnist import MNISTGuidanceEmbedder

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
    embedder = MNISTGuidanceEmbedder(model.backbone.d_model)

    print(f"num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.load_state_dict(t.load("weights/model_with_guidance_2.pth"))
    embedder.load_state_dict(t.load("weights/guidance_embedder_2.pth"))

    time_steps = t.tensor([15])
    with t.no_grad():
        for i, (x, y) in enumerate(data):
            x = model.forward_process(x,time_steps)
            if i > 7:
                break
            plt.imshow(x[0].squeeze().detach().numpy(), cmap='gray')
            plt.savefig(f"images/noised_num{y} iter{i}.png")
            plt.clf()
            one_hot_labels = nums_to_one_hot(t.Tensor([y]), 10)
            x = model.forward(x, time_steps, guidance=embedder(one_hot_labels, 0.0))
            plt.imshow(x[0].squeeze().detach().numpy(), cmap='gray')
            plt.savefig(f"images/denoised_num{y} iter{i}.png")
            plt.clf()
