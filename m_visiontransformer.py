import torch
from torch import Tensor, nn
import torch.nn.functional as F

class VisionTransformer(nn.Module):
  RESIDUAL_PLUS = 'plus'
  RESIDUAL_MERGE = 'merge'

  def __init__(self,
    d_model:int, reduction:int, embedding:int, nhead:int, layers:int,
    active_fn=F.silu, bias:bool=False, skips:bool=True, residual:str=None,
    device='cpu', dtype=torch.float32
  ) -> None:
    factory_kwargs = {
      'device': device,
      'dtype': dtype
    }
    super().__init__()

    self.d_model = d_model
    self.active_fn = active_fn
    self.residual = residual
    self.reduction = reduction
    self.skips = skips
    self.space_proj = embedding != d_model

    self.downs = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(d_model, d_model, 3, 1, 1, bias=bias, **factory_kwargs),
        nn.GroupNorm(1, d_model, **factory_kwargs),
      )
      for _ in range(reduction)
    ])

    self.ups = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(2*d_model if skips else d_model, d_model, 3, 1, 1, bias=bias, **factory_kwargs),
        nn.GroupNorm(1, d_model, **factory_kwargs),
      )
      for _ in range(reduction)
    ])

    self.residual_cnn = None
    if residual == 'merge':
      self.residual_cnn = nn.Sequential(
        nn.Conv2d(2 * d_model, d_model, 1, 1, 0, bias=bias, **factory_kwargs),
        nn.GroupNorm(1, d_model, **factory_kwargs)
      )

    self.encoder = nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model,
        nhead=nhead,
        activation=active_fn,
        batch_first=True,
        bias=bias,
        **factory_kwargs
      ),
      num_layers=layers,
      enable_nested_tensor=False
    )

    self.w_emb = nn.Linear(embedding, d_model, bias=bias, **factory_kwargs) if self.space_proj else nn.Identity()
    self.h_emb = nn.Linear(embedding, d_model, bias=bias, **factory_kwargs) if self.space_proj else nn.Identity()

  def __call__(self, X:Tensor, size_emb:tuple[Tensor, Tensor] = None, spacing:int=1) -> Tensor:
    return super().__call__(X, size_emb, spacing)

  def _apply_downsample(self, X:Tensor, memory:list[Tensor]=None) -> Tensor:
    for down in self.downs:
      X = F.max_pool2d(X, 2, 2)
      X = self.active_fn(down(X))
      if self.skips:
        memory.append(X)
    return X

  def _apply_upsample(self, X:Tensor, memory:list[Tensor]=None) -> Tensor:
    for index, up in enumerate(self.ups, start=1):
      if self.skips:
        X = torch.cat([X, memory[-index]], dim=1)
      X = self.active_fn(up(X))
      X = F.interpolate(X, scale_factor=2, mode='nearest')
    return X

  def _apply_size_embedding(self, X:Tensor, size_emb:tuple[Tensor, Tensor], spacing:int) -> Tensor:
    if size_emb is not None:
      pe_w, pe_h = size_emb
      pe_w = self.active_fn(self.w_emb(pe_w[::spacing,:]).moveaxis(0, -1))
      pe_h = self.active_fn(self.h_emb(pe_h[::spacing,:]).moveaxis(0, -1))
      X = X + pe_w[None, :, None, :] + pe_h[None, :, :, None]
    return X

  def _apply_transformer_encoder(self, X:Tensor) -> Tensor:
    n, _, dh, dw = X.shape
    X = X.view(n, self.d_model, dh * dw).moveaxis(-1, 1)
    X:Tensor = self.encoder(X)
    X = X.view(n, dh, dw, self.d_model).moveaxis(-1, 1)
    return X

  def _apply_residual(self, X:Tensor, X_residual:Tensor) -> Tensor:
    if self.residual == 'merge':
      X = torch.cat([X, X_residual], dim=1)
      X = self.active_fn(self.residual_cnn(X))

    if self.residual == 'plus':
      X[:, ::2, :, :] += X_residual[:, ::2, :, :]

    return X

  def forward(self, X:Tensor, size_emb:tuple[Tensor, Tensor], spacing:int) -> Tensor:
    X_residual:Tensor = X

    memory:list[Tensor] = [] if self.skips else None

    X = self._apply_downsample(X, memory)
    X = self._apply_size_embedding(X, size_emb, spacing)

    X = self._apply_transformer_encoder(X)

    X = self._apply_upsample(X, memory)

    if self.skips:
      memory.clear()

    X = self._apply_residual(X, X_residual)

    return X

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import numpy as np

  from PIL import Image

  from m_embedding import SpaceEmbedding

  image = np.array(Image.open('postprocessing/img-0.jpg').convert('RGB'), dtype=np.float32)
  image = (image / 255) * 2 - 1
  image = image.transpose(2, 0, 1)[None, :, :, :].repeat(8, 0)

  reduction = 4
  active_fn = F.gelu
  embedding = 128
  d_model = 256
  nhead=4
  layers = 8
  bias=False
  skips = True
  residual = VisionTransformer.RESIDUAL_PLUS

  inc = nn.Conv2d(3, d_model, 3, 1, 1)
  re = nn.Conv2d(d_model, 3, 3, 1, 1)
  size_embeding = SpaceEmbedding(embedding, (8, 8))

  size = size_embeding()

  vt = VisionTransformer(
    d_model, reduction, embedding, nhead, layers,
    active_fn, bias, skips, residual
  )

  print(image.shape)

  X = Tensor(image)
  X = inc(X)

  Z = vt(X, size, 1)

  Z = re(Z)
  Z = Z.detach()

  plt.imshow(Z[0, :3].moveaxis(0, -1).numpy() * 0.5 + 0.5)
  plt.show()