import glob
import os

from PIL import ImageFilter, Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class SRCNNDataset(Dataset):
    def __init__(self, transform):
        super(SRCNNDataset, self).__init__()
        self.transform = transform
        self.paths = glob.glob(os.path.join('BSDS300/train', '*.jpg'))

    def __getitem__(self, index):
        path = self.paths[index]
        img, _, _ = Image.open(path).convert('YCbCr').split()
        x = img.filter(ImageFilter.GaussianBlur(2))
        x = self.transform(x)
        return x, self.transform(img)

    def __len__(self):
        return len(self.paths)
