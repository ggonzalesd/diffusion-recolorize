import torch
from torch import nn, Tensor

from m_utils import create_sin_cos_encoding
from m_unet import UNet
from m_embedding import StepEmbedding, SpaceEmbedding

class Recolorate(nn.Module):
  def __init__(self,
    steps=1000, embedding:int=256, w_size:int=128, h_size:int=128,
    device='cpu', dtype=torch.float32
  ) -> None:
    factory_kwargs = {
      'device': device,
      'dtype': dtype,
    }
    super().__init__()

    self.space_emb = SpaceEmbedding(embedding, (w_size, h_size), **factory_kwargs)
    self.step_emb = StepEmbedding(embedding, steps, **factory_kwargs)
    self.unet = UNet(embedding, **factory_kwargs)

  def __call__(self, X:Tensor, t:Tensor, src:Tensor=None) -> Tensor:
    return super().__call__(X, t, src)

  def forward(self, X:Tensor, t:Tensor, src:Tensor) -> Tensor:

    step = self.step_emb(t)
    space = self.space_emb()

    X = self.unet(X, step, space, src)

    return X

if __name__ == '__main__':
  X = torch.randn(1, 3, 128, 128)
  src = X.mean(dim=1, keepdim=True)
  t = torch.randint(low=0, high=1000, size=(1,))

  model = Recolorate(1000, 256, 8, 8)

  predicted = model(X, t, src)

  print(predicted.shape)
