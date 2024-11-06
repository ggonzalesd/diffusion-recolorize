import torch
from torch import nn, Tensor
import torch.nn.functional as F

from m_utils import batch_image_join, batch_image_split, batch_sequential_to_batch_image, batch_image_to_batch_sequential, create_sin_cos_encoding

class Attention(nn.Module):
  def __init__(
    self,
    d_model:int, heads:int, layers:int=1, splits:int=1, spacing:int=1, size_emb:int=256, feedforward:int=512,
    active_fn=F.silu,
    device='cpu', dtype=torch.float32
  ):
    factory_kwargs = { 'device': device, 'dtype': dtype }
    super().__init__()
    self.d_model = d_model
    self.splits = splits
    self.spacing = spacing

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
    if size_emb is not None:
      pe_w, pe_h = size_emb
      pe_w = self.w_emb(pe_w[::self.spacing,:]).swapaxes(0, -1)
      pe_h = self.h_emb(pe_h[::self.spacing,:]).swapaxes(0, -1)

      X = X + pe_w[None, :, None, :] + pe_h[None, :, :, None]

    if self.splits > 1:
      X = batch_image_split(X, self.splits)
    X, sw, sh = batch_image_to_batch_sequential(X)

    X_ln = self.encoder(X)

    X_ln = batch_sequential_to_batch_image(X_ln, sw, sh)
    if self.splits > 1:
      X_ln = batch_image_join(X_ln, self.splits)

    return X_ln

if __name__ == '__main__':

  DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  EMBEDDING = 256
  STEPS = 1000
  BATCH = 32
  W_SIZE = 128
  H_SIZE = 128

  w_space = torch.arange(start=0, end=W_SIZE, requires_grad=False, dtype=torch.long)
  h_space = torch.arange(start=W_SIZE, end=W_SIZE+H_SIZE, requires_grad=False, dtype=torch.long)

  factory_kwargs = {
    'device': DEVICE,
    'dtype': torch.float32,
  }

  w_weight = create_sin_cos_encoding(EMBEDDING, W_SIZE)
  h_weight = create_sin_cos_encoding(EMBEDDING, H_SIZE)
  weight = torch.cat((w_weight, h_weight), dim=0)
  space_emb = nn.Embedding(num_embeddings=W_SIZE + H_SIZE, embedding_dim=EMBEDDING, _weight=weight)

  attention = Attention(
    d_model=128,
    heads=8,
    layers=1,
    splits=2,
    spacing=4,
    size_emb=EMBEDDING
  )

  X = torch.randn(BATCH, 128, 32, 32)
  w_space, h_space = space_emb(w_space), space_emb(h_space)

  print(w_space.shape)

  Z = attention(X, (w_space, h_space))

  print('X', X.shape)
  print('Z', Z.shape)
