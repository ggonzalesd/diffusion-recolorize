import torch
from torch import nn, Tensor
import torch.nn.functional as F

from m_convolutional import Convolutional
from m_utils import create_sin_cos_encoding
from m_down import DownBlock

class UpBlock(nn.Module):
  def __init__(self,
    in_ch:int, out_ch:int, step_embedding:int=512,
    active_fn=F.silu,
    device='cpu', dtype=torch.float32
  )->None:
    factory_kwargs = { 'device': device, 'dtype': dtype }
    super().__init__()

    self.active_fn = active_fn

    self.up = nn.Upsample(scale_factor=2, mode='nearest')
    self.conv = nn.Sequential(
      Convolutional(in_ch, in_ch, residual=True, **factory_kwargs),
      Convolutional(in_ch, out_ch, residual=False, **factory_kwargs),
    )

    self.step_emb = nn.Linear(step_embedding, out_ch, **factory_kwargs)

  def __call__(self, X:Tensor, step:Tensor, skip:Tensor, src:Tensor=None) -> Tensor:
    return super().__call__(X, step, skip, src)

  def forward(self, X:Tensor, step:Tensor, skip:Tensor, src:Tensor) -> Tensor:
    X = self.up(X)

    if src is not None:
      X[:,::2,:,:] += src
    X = torch.cat([skip, X], dim=1)
    X = self.conv(X)

    step_emb = self.step_emb(self.active_fn(step))[:, :, None, None]\
      .repeat(1, 1, X.shape[-2], X.shape[-1])

    return X + step_emb

if __name__ == '__main__':
  DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  EMBEDDING = 256
  STEPS = 1000
  BATCH = 32

  factory_kwargs = {
    'device': DEVICE,
    'dtype': torch.float32,
  }

  step_emb = nn.Embedding(STEPS, EMBEDDING, _weight=create_sin_cos_encoding(EMBEDDING, STEPS), **factory_kwargs)

  down = DownBlock(32, 64, EMBEDDING)
  up = UpBlock(64 + 32, 32, EMBEDDING)

  t = torch.randint(low=0, high=STEPS, size=(BATCH,), dtype=torch.long, device=DEVICE)
  src = torch.randn(BATCH, 1, 64, 64, **factory_kwargs)
  X = torch.randn(BATCH, 32, 64, 64, **factory_kwargs)

  t = step_emb(t)
  Z = down(X, t, src)
  A = up(Z, t, X, src)

  print('t', t.shape)
  print('src', src.shape)
  print('X', X.shape)
  print('Z', Z.shape)
  print('A', A.shape)
