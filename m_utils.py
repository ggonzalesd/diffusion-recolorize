import torch, math
from torch import Tensor

def create_sin_cos_encoding(d_model:int, max_length:int, dtype=torch.float32, device='cpu') -> Tensor:
  factory_kwargs = {
    'dtype': dtype,
    'device': device,
  }

  pe = torch.zeros(max_length, d_model, **factory_kwargs)
  position = torch.arange(0, max_length, **factory_kwargs).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, d_model, 2, **factory_kwargs) * -(math.log(1e+4) / d_model))

  pe[:, 0::2] = (position * div_term).sin()
  pe[:, 1::2] = (position * div_term).cos()

  return pe

def batch_image_split(batch:Tensor, splits:int) -> Tensor:
  sw = batch.shape[-2] // splits
  sh = batch.shape[-1] // splits

  return torch.stack([
    batch[:, :, j*sw:(j+1)*sw, i*sh:(i+1)*sh]
      for i in range(splits)
      for j in range(splits)
  ], dim=1).view(-1, batch.shape[1], sw, sh)

def batch_image_join(batch:Tensor, splits:int) -> Tensor:
  X = batch.view(-1, splits, splits, batch.shape[1], batch.shape[2], batch.shape[3])

  X = torch.cat([X[:, i] for i in range(splits)], dim=-1)
  X = torch.cat([X[:, i] for i in range(splits)], dim=-2)

  return X

def batch_image_to_batch_sequential(X:Tensor) -> Tensor:
  sw, sh = X.shape[-2], X.shape[-1]
  return X.view(X.shape[0], X.shape[1], -1).swapaxes(-1, 1), sw, sh

def batch_sequential_to_batch_image(X:Tensor, sw:int, sh:int) -> Tensor:
  return X.swapaxes(-1, 1).view(X.shape[0], X.shape[-1], sw, sh)

import torch, math
from torch import Tensor

def linear_scheduler(steps:int=1000, device='cpu', dtype=torch.float32):
  beta = torch.linspace(1e-4, 2e-2, steps, device=device, dtype=dtype)
  alpha = 1.0 - beta
  alpha_hat = torch.cumprod(alpha, dim=0)

  return alpha, beta, alpha_hat

def cosine_scheduler(steps:int=1000, device='cpu', dtype=torch.float32):
  s = 8e-3
  t = torch.linspace(0, steps, steps + 1, device=device, dtype=dtype)
  alpha_hat = torch.cos((t / steps + s) / (1 + s) * math.pi * 0.5) ** 2
  alpha_hat = alpha_hat / alpha_hat[0]
  beta = torch.clamp(1.0 - (alpha_hat[1:] / alpha_hat[:-1]), 0.0001, 0.9999)
  alpha = 1.0 - beta

  return alpha, beta, alpha_hat

def noise_image(X:Tensor, t:Tensor, s:tuple[Tensor, Tensor, Tensor], device='cpu', dtype=torch.float32) -> tuple[Tensor, Tensor]:
  _, _, alpha_hat = s

  sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
  sqrt_one_alpha_hat = torch.sqrt(1.0 - alpha_hat[t])[:, None, None, None]
  e = torch.randn_like(X, device=device, dtype=dtype)
  return sqrt_alpha_hat * X + sqrt_one_alpha_hat * e, e

def sample_timesteps(n: int, steps:int=1000, device='cpu') -> Tensor:
  return torch.randint(low=1, high=steps, size=(n,), device=device)

def batch_image_to_batch_sequential(X:Tensor) -> Tensor:
  sw, sh = X.shape[-2], X.shape[-1]
  return X.view(X.shape[0], X.shape[1], -1).swapaxes(-1, 1), sw, sh

def batch_sequential_to_batch_image(X:Tensor, sw:int, sh:int) -> Tensor:
  return X.swapaxes(-1, 1).view(X.shape[0], X.shape[-1], sw, sh)

if __name__ == '__main__':
  import os, glob
  from PIL import Image
  import numpy as np
  import torch

  import matplotlib.pyplot as plt

  image = Image.open('postprocessing/img-0.jpg').convert('RGB')
  image = torch.Tensor(np.array(image, dtype=np.float32) / 255).unsqueeze(0).moveaxis(-1, 1)

  sequence = image.view(1, 3, -1).swapaxes(-1, 1)

  restore = sequence.swapaxes(-1, 1).view(1, 3, 128, 128)

  plt.subplot(1, 2, 1)
  plt.imshow(image[0, :3].moveaxis(0, -1).detach().cpu().numpy())
  plt.subplot(1, 2, 2)
  plt.imshow(restore[0, :3].moveaxis(0, -1).detach().cpu().numpy())
  plt.show()
