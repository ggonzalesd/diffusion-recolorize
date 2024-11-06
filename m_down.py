import torch
from torch import nn, Tensor
import torch.nn.functional as F

from m_convolutional import Convolutional
from m_utils import create_sin_cos_encoding

class DownBlock(nn.Module):
  def __init__(self,
    in_ch:int, out_ch:int, step_embedding:int=512,
    active_fn=F.silu,
    device='cpu', dtype=torch.float32
  ) -> None:
    factory_kwargs = { 'device': device, 'dtype': dtype }
    super().__init__()

    self.active_fn = active_fn

    self.conv = nn.Sequential(
      Convolutional(in_ch, in_ch, residual=True, active_fn=active_fn, **factory_kwargs),
      Convolutional(in_ch, out_ch, residual=False, active_fn=active_fn, **factory_kwargs),
      nn.MaxPool2d(2),
    )

    self.step_emb = nn.Linear(step_embedding, out_ch, **factory_kwargs)

  def __call__(self, X:Tensor, step:Tensor, src:Tensor=None) -> Tensor:
    return super().__call__(X, step, src)

  def forward(self, X:Tensor, step:Tensor, src:Tensor=None) -> Tensor:
    X[:,::2,:,:] += src
    X = self.conv(X)

    step_emb = self.step_emb(self.active_fn(step))[:, :, None, None]\
      .repeat(1, 1, X.shape[-2], X.shape[-1])

    return X + step_emb

if __name__ == '__main__':

  DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  CATEGORIES = 10
  EMBEDDING = 256
  STEPS = 1000
  BATCH = 32

  factory_kwargs = {
    'device': DEVICE,
    'dtype': torch.float32,
  }

  step_emb = nn.Embedding(STEPS, EMBEDDING, _weight=create_sin_cos_encoding(EMBEDDING, STEPS), **factory_kwargs)

  down = DownBlock(3, 64, EMBEDDING)

  X = torch.randn(BATCH, 3, 64, 64, **factory_kwargs)
  t = torch.randint(low=0, high=STEPS, size=(BATCH,), dtype=torch.long, device=DEVICE)

  t = step_emb(t)
  src = torch.randn(BATCH, 1, 64, 64)
  Z = down(X, t, src)

  print('t', t.shape)
  print('src', src.shape)
  print('X', X.shape)
  print('Z', Z.shape)
