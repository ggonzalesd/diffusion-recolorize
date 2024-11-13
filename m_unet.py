import torch
from torch import Tensor, nn

import torch.nn.functional as F

from m_convolutional import Convolutional
from m_down import DownBlock
from m_up import UpBlock
from m_visiontransformer import VisionTransformer

class UNet(nn.Module):
  def __init__(self, embedding:int=256, active_fn=F.silu, device='cpu', dtype=torch.float32):
    factory_kwargs = {
      'device': device,
      'dtype': dtype
    }
    super().__init__()

    self.active_fn = active_fn

    self.inc1 = Convolutional(
      in_ch=3, out_ch=64, active_fn=active_fn, **factory_kwargs
    )

    self.down1 = DownBlock(
      in_ch=64, out_ch=128, step_embedding=embedding, active_fn=active_fn, **factory_kwargs
    ) # 64
    self.sa1 = VisionTransformer(
      d_model=128, reduction=3, embedding=embedding, nhead=4, layers=4,
      active_fn=active_fn, bias=False, skips=True, residual=VisionTransformer.RESIDUAL_MERGE,
      **factory_kwargs
    )

    self.down2 = DownBlock(
      in_ch=128, out_ch=256, step_embedding=embedding, active_fn=active_fn, **factory_kwargs
    ) # 32
    self.sa2 = VisionTransformer(
      d_model=256, reduction=2, embedding=embedding, nhead=4, layers=4,
      active_fn=active_fn, bias=False, skips=True, residual=VisionTransformer.RESIDUAL_MERGE,
      **factory_kwargs
    )

    self.down3 = DownBlock(
      in_ch=256, out_ch=256, step_embedding=embedding, active_fn=active_fn, **factory_kwargs
    ) # 16
    self.sa3 = VisionTransformer(
      d_model=256, reduction=1, embedding=embedding, nhead=4, layers=4,
      active_fn=active_fn, bias=False, skips=True, residual=VisionTransformer.RESIDUAL_MERGE,
      **factory_kwargs
    )

    self.bot1 = Convolutional(256, 512, active_fn=active_fn, **factory_kwargs)
    self.bot2 = Convolutional(512, 512, active_fn=active_fn, residual=True, **factory_kwargs)
    self.bot3 = Convolutional(512, 256, active_fn=active_fn, **factory_kwargs)

    self.up1 = UpBlock(
      in_ch=256 + 256, out_ch=256, step_embedding=embedding, active_fn=active_fn, **factory_kwargs
    )
    self.sa4 = VisionTransformer(
      d_model=256, reduction=2, embedding=embedding, nhead=4, layers=4,
      active_fn=active_fn, bias=False, skips=True, residual=VisionTransformer.RESIDUAL_MERGE,
      **factory_kwargs
    )

    self.up2 = UpBlock(
      in_ch=128 + 2 * 128, out_ch=128, step_embedding=embedding, active_fn=active_fn, **factory_kwargs
    )
    self.sa5 = VisionTransformer(
      d_model=128, reduction=3, embedding=embedding, nhead=4, layers=4,
      active_fn=active_fn, bias=False, skips=True, residual=VisionTransformer.RESIDUAL_MERGE,
      **factory_kwargs
    )

    self.up3 = UpBlock(
      in_ch=64 + 2*64, out_ch=64, step_embedding=embedding, active_fn=active_fn, **factory_kwargs
    )
    self.sa6 = VisionTransformer(
      d_model=64, reduction=4, embedding=embedding, nhead=4, layers=4,
      active_fn=active_fn, bias=False, skips=True, residual=VisionTransformer.RESIDUAL_MERGE,
      **factory_kwargs
    )

    self.out = nn.Conv2d(64, 3, 1, 1, bias=False, **factory_kwargs)

  def __call__(self, X:Tensor, step:Tensor, s:tuple[Tensor, Tensor], src:Tensor=None) -> Tensor:
    return super().__call__(X, step, s, src)

  def forward(self, X:Tensor, step:Tensor, s:tuple[Tensor, Tensor], src:Tensor) -> Tensor:
    Z1 = self.inc1(X) # 64 128 128
    Z2 = self.down1(Z1, step, src) # 128 64 64
    Z2 = self.sa1(Z2, s) # 128 64 64
    Z3 = self.down2(Z2, step, F.max_pool2d(src, 2) if src is not None else None) # 256 32 32
    Z3 = self.sa2(Z3, s) # 256 32 32
    Z4 = self.down3(Z3, step, F.max_pool2d(src, 4) if src is not None else None) # 256 16 16
    Z4 = self.sa3(Z4, s) # 256 16 16

    A = self.bot1(Z4) # 256 16 16
    A = self.bot2(A) # 256 16 16
    A = self.bot3(A) # 256 16 16

    A = self.up1(A, step, Z3, F.max_pool2d(src, 4) if src is not None else None) # 256 32 32
    A = self.sa4(A, s) # 256 32 32

    A = self.up2(A, step, Z2, F.max_pool2d(src, 2) if src is not None else None) # 128 64 64
    A = self.sa5(A, s)

    A = self.up3(A, step, Z1, src) # 64 128 128
    A = self.sa6(A, s)

    A = self.out(A)

    return A

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import numpy as np

  from PIL import Image

  from m_embedding import SpaceEmbedding
  from m_embedding import StepEmbedding
  from m_visiontransformer import VisionTransformer

  image = np.array(Image.open('postprocessing/img-0.jpg').convert('RGB'), dtype=np.float32)
  image = (image / 255) * 2 - 1
  image = image.transpose(2, 0, 1)[None, :, :, :].repeat(2, 0)

  reduction = 4
  active_fn = F.gelu
  embedding = 128
  d_model = 256
  nhead=4
  layers = 8
  bias=False
  skips = True
  residual = VisionTransformer.RESIDUAL_PLUS

  space_emb = SpaceEmbedding(embedding, (8, 8))
  step_emb = StepEmbedding(embedding, 1000)
  unet = UNet(embedding, F.silu)

  X = Tensor(image)
  t = torch.randint(low=0, high=1000, size=(X.shape[0],))

  step = step_emb(t)
  space = space_emb()

  X = unet(torch.randn_like(X) * 0.05 + X * 0.95, step, space, X.mean(1, keepdim=True))

  print(X.shape)

  plt.imshow(X.detach()[0, :3].moveaxis(0, -1).numpy() * 0.5 + 0.5)
  plt.show()