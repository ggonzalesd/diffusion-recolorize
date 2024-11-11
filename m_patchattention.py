import torch
from torch import nn, Tensor
import torch.nn.functional as F

from m_utils import sequential_features_to_image, image_to_sequential_features

class PathAttention(nn.Module):
  def __init__(
    self,
    channels:int, d_model:int, heads:int, layers:int=1,
    splits:int=1, spacing:int=1,
    size_emb:int=256, feedforward:int=512,
    active_fn=F.silu,
    device='cpu', dtype=torch.float32
  ):
    factory_kwargs = { 'device': device, 'dtype': dtype }
    super().__init__()
    self.d_model = d_model
    self.splits = splits
    self.spacing = spacing
    self.channels = channels
    self.active_fn = active_fn

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

    self.in_projection = nn.Linear(channels * splits * splits, d_model, bias=False, **factory_kwargs)
    self.out_projection = nn.Linear(d_model, channels * splits * splits, bias=False, **factory_kwargs)

    self.w_emb = nn.Linear(size_emb, d_model, bias=False, **factory_kwargs)
    self.h_emb = nn.Linear(size_emb, d_model, bias=False, **factory_kwargs)

    self.norm_out = nn.GroupNorm(1, d_model, **factory_kwargs)

  def __call__(self, X:Tensor, size_emb:tuple[Tensor, Tensor]=None) -> Tensor:
    # X: [N, C, W, H]
    # size_emb: ([ W_SIZE, E ], [ H_SIZE, E ])
    return super().__call__(X, size_emb)

  def forward(self, X:Tensor, size_emb:tuple[Tensor, Tensor]) -> Tensor:
    Z:Tensor = X

    # Space Embedding
    if size_emb is not None:
      pe_w, pe_h = size_emb

      pe_w = self.w_emb(pe_w[::self.spacing,:]).moveaxis(0, -1)
      pe_h = self.h_emb(pe_h[::self.spacing,:]).moveaxis(0, -1)

      X = X + pe_w[None, :, None, :] + pe_h[None, :, :, None]

    # Split Image if is needed
    X, ss, features = image_to_sequential_features(X, self.splits)
    X:Tensor = self.active_fn(self.in_projection(X))

    # Apply Encoder Transformer
    X:Tensor = self.encoder(X)

    # Convert Sequence into Image
    X:Tensor = self.active_fn(self.out_projection(X))
    X = sequential_features_to_image(X, self.splits, ss, features)

    # X[:, ::2, :, :] += Z[:, ::2, :, :]

    return X

if __name__ == '__main__':
  from m_embedding import SpaceEmbedding
  import matplotlib.pyplot as plt
  import time

  channels = 32
  groups = 4

  X = torch.randn(16, channels, 128, 128)
  attention = PathAttention(
    channels,
    channels,
    heads=8,
    layers=2,
    splits=8,
    spacing=1,
    size_emb=128
  )
  size_embeding = SpaceEmbedding(128, (128, 128))

  out_projection = nn.Conv2d(channels, 3, 1, 1, bias=False)

  t = time.time()
  size = size_embeding()
  Z = attention(X, size)
  t = time.time() - t
  A:Tensor = out_projection(Z).clip(0, 1)

  print('X', X.shape)
  print('Z', Z.shape)
  print('t', t)
  print('Z:', Z.min().item(), Z.max().item(), Z.std().item(), Z.mean().item())

  plt.subplot(1, 3, 1)
  plt.imshow(X[0][:3].moveaxis(0, -1).detach().cpu().numpy())
  plt.subplot(1, 3, 2)
  plt.imshow(Z[0][:3].moveaxis(0, -1).detach().cpu().numpy())
  plt.subplot(1, 3, 3)
  plt.imshow(A[0][:3].moveaxis(0, -1).detach().cpu().numpy())
  plt.show()
