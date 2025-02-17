import torch as t
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from time_step_sampler import TimeStepSampler
import torch.nn as nn
from diffusion import Diffusion
import torchvision
import torchvision.transforms as transforms

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

def nums_to_one_hot(nums: t.Tensor, d_input: int) -> t.Tensor:
    """
    :param nums: (batch_size,)
    :param d_input: int
    :return: (batch_size, d_input)
    """
    one_hots = t.zeros(nums.shape[0], d_input)
    one_hots.scatter_(1, nums.long().unsqueeze(1), 1)
    return one_hots

# Wrote this for MNIST to turn 1 hot encoded labels into guidance embeddings
class MNISTGuidanceEmbedder(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.d_input = 10
        self.d_embedding = d_embedding
        self.projection = nn.Linear(10, d_embedding)
        self.l2 = nn.Linear(d_embedding, d_embedding)
        self.relu = nn.ReLU()
        # for classifier free guidance
        self.null_guidance = nn.Parameter(t.zeros(self.d_input), requires_grad=True)

    def forward(self, guidance: t.Tensor, guidance_free_prob: float=0.1) -> t.Tensor:
        """
        :param guidance: (batch_size, d_guidance)
        :param guidance_free_prob: float
        :return: (batch_size, d_embedding)
        """
        # randomly choose indices to replace batch samples with null guidance
        null_guidance_indices = t.rand(guidance.shape[0]) < guidance_free_prob
        guidance[null_guidance_indices] = self.null_guidance
        guidance = self.projection(guidance)
        guidance = self.relu(guidance)
        guidance = self.l2(guidance)
        return guidance


def train(model,
guidance_embedder: MNISTGuidanceEmbedder,
data: Dataset, 
epochs: int=1, 
batch_size: int=64, 
print_intervals: int=1, 
debug: bool=False, 
batches: int=float('inf'), 
time_steps: int=None, 
lr: float=1e-4,
use_importance_sampling: bool=True,
guidance_free_prob: float=0.1):
    data = DataLoader(data, batch_size=batch_size, shuffle=True)
    
    model = model.to(DEVICE)
    guidance_embedder = MNISTGuidanceEmbedder(model.backbone.d_model).to(DEVICE)

    optimizer = t.optim.Adam(
        list(model.parameters()) + list(guidance_embedder.parameters()),
        lr=lr
    )
    time_step_sampler = TimeStepSampler(model.training_time_steps, use_importance_sampling=use_importance_sampling)
    model.losses = []

    for epoch in range(epochs):
        for i ,(batch, labels) in enumerate(data):
            
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)

            if i >= batches:
                break

            time_steps = time_step_sampler.sample_time_steps(batch.shape[0]).to(DEVICE)
            one_hot_labels = nums_to_one_hot(labels.long(), 10)
            guidance = guidance_embedder(one_hot_labels, guidance_free_prob)

            noisy_data = model.forward_process(batch, time_steps)
            predicted_batch = model(noisy_data, time_steps, guidance=guidance)
            batch_losses = t.nn.MSELoss(reduction='none')(predicted_batch, batch).mean(dim=[1, 2, 3])
            loss = batch_losses.mean()
            
            if use_importance_sampling:
                time_step_sampler.update_losses(time_steps.detach().numpy(), batch_losses.detach().numpy())
            
            model.losses.append(loss.detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_intervals == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{batches}], Loss: {loss.item():.4f}")

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
    model.load_state_dict(t.load("weights/model_with_guidance_2.pth"))
    embedder.load_state_dict(t.load("weights/guidance_embedder_2.pth"))
    train(
        model,embedder, data, epochs=2, 
        batch_size=64, print_intervals=5, 
        debug=True, lr=2e-5
    )

    t.save(model.state_dict(), "weights/model_with_guidance_2.pth")
    t.save(embedder.state_dict(), "weights/guidance_embedder_2.pth")