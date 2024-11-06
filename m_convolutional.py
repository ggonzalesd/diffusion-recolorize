import torch
from torch import nn, Tensor
import torch.nn.functional as F

class Convolutional(nn.Module):
  def __init__(self,
    in_ch:int, out_ch:int, mid_ch:int=None,
    residual:bool=False, active_fn=F.silu,
    device='cpu', dtype=torch.float32
  ) -> None:
    factory_kwargs = { 'device': device, 'dtype': dtype }
    super().__init__()

    self.residual = residual
    self.f = active_fn

    if mid_ch == None:
      mid_ch = out_ch

    self.conv1 = nn.Sequential(
      nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False, **factory_kwargs),
      nn.GroupNorm(1, mid_ch, **factory_kwargs),
    )

    self.conv2 = nn.Sequential(
      nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False, **factory_kwargs),
      nn.GroupNorm(1, out_ch, **factory_kwargs)
    )

  def __call__(self, X:Tensor) -> Tensor:
    return super().__call__(X)

  def forward(self, X:Tensor) -> Tensor:
    Z = self.f(self.conv1(X))
    Z = self.conv2(Z)

    if self.residual:
      return self.f(X + Z)

    return Z

if __name__ == '__main__':
  conv = Convolutional(
    in_ch=3,
    out_ch=64,
    mid_ch=128,
  )

  X = torch.randn(8, 3, 64, 64)

  Z = conv(X)

  print('X', X.shape)
  print('Z', Z.shape)
