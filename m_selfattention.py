import torch
from torch import nn, Tensor
import torch.nn.functional as F

from m_utils import batch_image_join, batch_image_split, batch_sequential_to_batch_image, batch_image_to_batch_sequential

class SelfAttention(nn.Module):
  def __init__(
    self,
    d_model:int, heads:int, layers:int=1,
    splits:int=1, spacing:int=1, reduction:int=0,
    size_emb:int=256, feedforward:int=512,
    active_fn=F.silu,
    device='cpu', dtype=torch.float32
  ):
    factory_kwargs = { 'device': device, 'dtype': dtype }
    super().__init__()
    self.d_model = d_model
    self.splits = splits
    self.spacing = spacing
    self.reduction = reduction
    self.active_fn = active_fn

    downs = []
    for _ in range(reduction):
      downs.append(nn.Conv2d(d_model, d_model, 3, 1, 1, bias=False))
      downs.append(nn.GroupNorm(1, d_model, **factory_kwargs))
      downs.append(nn.MaxPool2d(2))
    self.downs = nn.ModuleList(downs) if reduction > 0 else []

    ups = []
    for _ in range(reduction):
      ups.append(nn.Upsample(scale_factor=2, mode='nearest'))
      ups.append(nn.Conv2d(d_model, d_model, 3, 1, 1, bias=False))
      ups.append(nn.GroupNorm(1, d_model, **factory_kwargs))
    self.ups = nn.ModuleList(ups) if reduction > 0 else []

    self.encoder = nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model,
        nhead=heads,
        activation=active_fn,
        batch_first=True,
        bias=False,
        norm_first=True,
        dim_feedforward=feedforward,
        **factory_kwargs
      ),
      enable_nested_tensor=False,
      num_layers=layers,
    )

    self.w_emb = nn.Linear(size_emb, d_model)
    self.h_emb = nn.Linear(size_emb, d_model)

  def __call__(self, X:Tensor, size_emb:tuple[Tensor, Tensor]=None) -> Tensor:
    # X: [N, C, W, H]
    # size_emb: ([ W_SIZE, E ], [ H_SIZE, E ])
    return super().__call__(X, size_emb)

  def forward(self, X:Tensor, size_emb:tuple[Tensor, Tensor]) -> Tensor:
    Z:Tensor = X

    # Reduce Size of Image
    for index, down in enumerate(self.downs):
      X:Tensor = down(X)
      if index % 3 == 1:
        X:Tensor = self.active_fn(X)

    # Space Embedding
    if size_emb is not None:
      pe_w, pe_h = size_emb
      pe_w = self.w_emb(pe_w[::self.spacing,:]).moveaxis(0, -1)
      pe_h = self.h_emb(pe_h[::self.spacing,:]).moveaxis(0, -1)

      X = X + pe_w[None, :, None, :] + pe_h[None, :, :, None]

    # Split Image if is needed
    if self.splits > 1:
      X = batch_image_split(X, self.splits)
    # Convert Image into Sequence
    X, sw, sh = batch_image_to_batch_sequential(X)

    # Apply Encoder Transformer
    X:Tensor = self.encoder(X)

    # Convert Sequence into Image
    X = batch_sequential_to_batch_image(X, sw, sh)
    # Rebuild Image from splits
    if self.splits > 1:
      X = batch_image_join(X, self.splits)

    # Reisze Image
    for index, up in enumerate(self.ups):
      X:Tensor = up(X)
      if index % 3 == 2:
        X:Tensor = self.active_fn(X)

    X[:, ::2, :, :] += Z[:, ::2, :, :]

    return X

if __name__ == '__main__':
  from m_embedding import SpaceEmbedding
  import matplotlib.pyplot as plt
  import time

  channels = 32
  groups = 4
  reduction = 3

  X = torch.randn(16, channels, 128, 128)
  attention = SelfAttention(channels, 4, 4, 1, 2, 3, 128)
  size_embeding = SpaceEmbedding(128, (32, 32))

  t = time.time()
  size = size_embeding()
  Z = attention(X, size)
  t = time.time() - t

  print('X', X.shape)
  print('Z', Z.shape)
  print('t', t)

  plt.subplot(1, 2, 1)
  plt.imshow(X[0][:3].moveaxis(0, -1).detach().cpu().numpy())
  plt.subplot(1, 2, 2)
  plt.imshow(Z[0][:3].moveaxis(0, -1).detach().cpu().numpy())
  plt.show()
