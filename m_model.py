import torch
from torch import nn, Tensor

from m_utils import create_sin_cos_encoding
from m_unet import UNet

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

    w_weight = create_sin_cos_encoding(embedding, w_size, **factory_kwargs)
    h_weight = create_sin_cos_encoding(embedding, h_size, **factory_kwargs)
    weight = torch.cat((w_weight, h_weight), dim=0)
    self.space_emb = nn.Embedding(
      num_embeddings=w_size + h_size, embedding_dim=embedding, _weight=weight, **factory_kwargs
    )

    self.step_emb = nn.Embedding(
      num_embeddings=steps,
      embedding_dim=embedding,
      _weight=create_sin_cos_encoding(embedding, steps, **factory_kwargs),
      **factory_kwargs
    )

    self.unet = UNet(embedding, **factory_kwargs)

    self.register_buffer(
      'w_space',
      torch.arange(start=0, end=w_size, requires_grad=False, dtype=torch.long)
    )
    self.register_buffer(
      'h_space',
      torch.arange(start=w_size, end=w_size+h_size, requires_grad=False, dtype=torch.long)
    )

  def __call__(self, X:Tensor, t:Tensor, src:Tensor) -> Tensor:
    return super().__call__(X, t, src)

  def forward(self, X:Tensor, t:Tensor, src:Tensor) -> Tensor:
    w_space, h_space = self.get_buffer('w_space'), self.get_buffer('h_space')

    step = self.step_emb(t)
    space = self.space_emb(w_space), self.space_emb(h_space)

    X = self.unet(X, step, space, src)

    return X
