import torch
from torch import Tensor, nn

import torch.nn.functional as F

from m_convolutional import Convolutional
from m_down import DownBlock
from m_up import UpBlock
from m_selfattention import SelfAttention

class UNet(nn.Module):
  def __init__(self, embedding:int=256, device='cpu', dtype=torch.float32):
    factory_kwargs = {
      'device': device,
      'dtype': dtype
    }

    super().__init__()

    self.inc1 = Convolutional(
      in_ch=3, out_ch=64, **factory_kwargs
    )

    self.down1 = DownBlock(
      in_ch=64, out_ch=128, step_embedding=embedding, **factory_kwargs
    )
    self.sa1 = SelfAttention(
      d_model=128, heads=16, layers=4, splits=1, spacing=1, reduction=2, size_emb=embedding, **factory_kwargs
    )

    self.down2 = DownBlock(
      in_ch=128, out_ch=256, step_embedding=embedding, **factory_kwargs
    )
    self.sa2 = SelfAttention(
      d_model=256, heads=32, layers=4, splits=1, spacing=1, reduction=1, size_emb=embedding, **factory_kwargs
    )

    self.down3 = DownBlock(
      in_ch=256, out_ch=256, step_embedding=embedding, **factory_kwargs
    )
    self.sa3 = SelfAttention(
      d_model=256, heads=32, layers=4, splits=1, spacing=1, reduction=0, size_emb=embedding, **factory_kwargs
    )

    self.bot1 = Convolutional(256, 512, **factory_kwargs)
    self.bot2 = Convolutional(512, 512, residual=True, **factory_kwargs)
    self.bot3 = Convolutional(512, 256, **factory_kwargs)

    self.up1 = UpBlock(
      in_ch=256 + 256, out_ch=256, step_embedding=embedding, **factory_kwargs
    )
    self.sa4 = SelfAttention(
      d_model=256, heads=32, layers=4, splits=1, spacing=1, reduction=1, size_emb=embedding, **factory_kwargs
    )

    self.up2 = UpBlock(
      in_ch=128 + 2 * 128, out_ch=128, step_embedding=embedding, **factory_kwargs
    )
    self.sa5 = SelfAttention(
      d_model=128, heads=32, layers=4, splits=1, spacing=1, reduction=2, size_emb=embedding, **factory_kwargs
    )

    self.up3 = UpBlock(
      in_ch=64 + 2*64, out_ch=64, step_embedding=embedding, **factory_kwargs
    )
    self.sa6 = SelfAttention(
      d_model=64, heads=16, layers=4, splits=1, spacing=1, reduction=3, size_emb=embedding, **factory_kwargs
    )

    self.out = nn.Conv2d(64, 3, 1, bias=False, **factory_kwargs)

  def __call__(self, X:Tensor, step:Tensor, s:tuple[Tensor, Tensor], src:Tensor) -> Tensor:
    return super().__call__(X, step, s, src)

  def forward(self, X:Tensor, step:Tensor, s:tuple[Tensor, Tensor], src:Tensor) -> Tensor:
    Z1 = self.inc1(X) # 64 128 128
    Z2 = self.down1(Z1, step, src) # 128 64 64
    Z2 = self.sa1(Z2, s) + Z2 # 128 64 64
    Z3 = self.down2(Z2, step, F.max_pool2d(src, 2)) # 256 32 32
    Z3 = self.sa2(Z3, s) + Z3 # 256 32 32
    Z4 = self.down3(Z3, step, F.max_pool2d(src, 4)) # 256 16 16
    Z4 = self.sa3(Z4, s) + Z4 # 256 16 16

    A = self.bot1(Z4) # 256 16 16
    A = self.bot2(A) # 256 16 16
    A = self.bot3(A) # 256 16 16

    A = self.up1(A, step, Z3, F.max_pool2d(src, 4)) # 256 32 32
    A = self.sa4(A, s) + A # 256 32 32

    A = self.up2(A, step, Z2, F.max_pool2d(src, 2)) # 128 64 64
    A = self.sa5(A, s) + A

    A = self.up3(A, step, Z1, src) # 64 128 128
    A = self.sa6(A, s) + A

    A = self.out(A)

    return A