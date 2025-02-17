import torch as t
from u_net import UNet
from diffusion_transformer import DiffusionTransformer
from typing import Union

class Diffusion(t.nn.Module):

    def __init__(self, 
    input_shape: tuple[int,...],
    use_importance_sampling: bool=True, 
    training_time_steps: int=1000,
    backbone: Union[UNet, DiffusionTransformer, None]=None,
    d_guidance_input: int=1):
        super().__init__()
        self.use_importance_sampling: bool = use_importance_sampling
        self.beta_start = 10 ** -4
        self.beta_stop = 10 ** -1
        self.training_time_steps = training_time_steps
        self.betas = t.linspace(self.beta_start, self.beta_stop, self.training_time_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = t.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = t.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = t.sqrt(1-self.alpha_bars)

        if backbone is None:
            self.backbone = self._get_default_backbone(input_shape)
        else:
            self.backbone = backbone

    @staticmethod
    def _get_default_backbone(input_shape: tuple[int,...]):
        return DiffusionTransformer(
            input_shape=input_shape, 
            num_heads=4, 
            num_layers=6, 
            d_model=128, 
            d_ff=256, 
            patch_size=4
        )


    def forward_process(self, x: t.Tensor, time_steps: t.Tensor) -> t.Tensor:
        """
        :param x: tensor with shape (batch, channels, d_image, d_image)
        :param time_steps: tensor with shape (batch,)
        """
        epsilon = t.normal(0, 1, x.shape)

        # Reshape alpha and beta terms to match image dimensions
        sqrt_alpha = self.sqrt_alpha_bars[time_steps].contiguous().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_bars[time_steps].contiguous().view(-1, 1, 1, 1)
        res = sqrt_alpha * x + sqrt_one_minus_alpha * epsilon  
        return res
    

    def forward(self, x, time_steps, guidance=None):
        """
        :param x: tensor with shape (batch, channels, d_image, d_image)
        :param time_steps: tensor with shape (batch, d_conditioning)
        :param guidance: tensor with shape (batch, d_conditioning)
        """
        return self.backbone(x, time_steps, guidance)
  

    def sample(self, sample_steps, batch_size: int=1, data_shape: tuple=(1, 28, 28)):
        """
        :param batch_size: int, number of samples to sample
        """
        self.eval()
        noised = t.randn(batch_size, *data_shape)
        with t.no_grad():
            for ts in range(sample_steps,0,-1):
                noised = self.forward(noised)
                # noised = sqrt_alpha * x + sqrt_one_minus_alpha * noise
                # noised - (sqrt_one_minus_alpha * noise)/sqrt_alpha = x
                # predicted_noise = predicted_image - noised
                # noised -= predicted_noise/self.sqrt_alpha_bars[ts-1]
        self.train()
        return t.clamp(noised, -1, 1)
  
