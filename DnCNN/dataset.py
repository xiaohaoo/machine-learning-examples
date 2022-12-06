import glob
import os
import os.path

import torch

import torch.utils.data as udata
from PIL import Image


class Dataset(udata.Dataset):
    def __init__(self, root, transform):
        super(Dataset, self).__init__()
        self.transform = transform
        self.images = glob.glob(os.path.join(root, '*.png'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        img = self.transform(img)
        noise = torch.randn(img.shape) * (25.0 / 255.0)
        return img + noise, img
