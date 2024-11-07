import torch
from torch import Tensor, nn

from m_utils import create_sin_cos_encoding

class SpaceEmbedding(nn.Module):
  def __init__(self, embedding:int, size:tuple[int, int], device='cpu', dtype=torch.float32):
    factory_kwargs = {
      'device': device,
      'dtype': dtype
    }
    super().__init__()

    w_size, h_size = size

    w_weight = create_sin_cos_encoding(embedding, w_size, **factory_kwargs)
    h_weight = create_sin_cos_encoding(embedding, h_size, **factory_kwargs)
    weight = torch.cat((w_weight, h_weight), dim=0)
    self.embedding = nn.Embedding(
      num_embeddings=w_size + h_size, embedding_dim=embedding, _weight=weight, **factory_kwargs
    )

    self.register_buffer(
      'w_space',
      torch.arange(start=0, end=w_size, requires_grad=False, dtype=torch.long)
    )
    self.register_buffer(
      'h_space',
      torch.arange(start=w_size, end=w_size+h_size, requires_grad=False, dtype=torch.long)
    )

  def __call__(self) -> Tensor:
    return super().__call__()

  def forward(self) -> Tensor:
    w_space, h_space = self.get_buffer('w_space'), self.get_buffer('h_space')
    space = self.embedding(w_space), self.embedding(h_space)
    return space

class StepEmbedding(nn.Module):
  def __init__(self, embedding=128, steps=1000, device='cpu', dtype=torch.float32):
    factory_kwargs = {
      'device': device,
      'dtype': dtype
    }
    super().__init__()

    self.embedding = nn.Embedding(
      num_embeddings=steps,
      embedding_dim=embedding,
      _weight=create_sin_cos_encoding(embedding, steps, **factory_kwargs),
      **factory_kwargs
    )

  def __call__(self, t:Tensor) -> Tensor:
    return super().__call__(t)

  def forward(self, t:Tensor) -> Tensor:
    step:Tensor = self.embedding(t)
    return step
