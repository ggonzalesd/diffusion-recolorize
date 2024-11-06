import numpy as np
import glob, os

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T

from PIL import Image

class LandspaceDataset(Dataset):
  def __init__(self, path:str, transform: T.Compose = None, device='cpu'):
    super().__init__()

    self.path = path
    self.device = device
    self.filenames = glob.glob(os.path.join(path, '*.jpg'))
    mean_std = Tensor(np.load(os.path.join(path, 'mean_std.npy')), device=device)[:, :, None, None]
    self.mean = mean_std[0]
    self.std = mean_std[1]
    self.transform = transform if transform is not None else T.ToTensor()

  def __repr__(self) -> str:
    return f'<LandscapeDataset len:{len(self)}>'

  def __len__(self) -> int:
    return len(self.filenames)

  def __getitem__(self, index:int) -> Tensor:
    filename = self.filenames[index]

    image = Image.open(filename).convert('RGB')
    image = self.transform(image)
    image = (image - self.mean) / self.std

    return image, image.mean(0, keepdim=True)