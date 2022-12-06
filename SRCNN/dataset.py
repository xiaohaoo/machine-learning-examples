import glob
import os

import numpy as np
from PIL import ImageFilter, Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class SRCNNDataset(Dataset):
    def __init__(self, transform, train=True):
        super(SRCNNDataset, self).__init__()
        self.transform = transform
        if train:
            self.paths = glob.glob(os.path.join('BSDS300/train', '*.jpg'))
        else:
            self.paths = glob.glob(os.path.join('BSDS300/test', '*.jpg'))

    def __getitem__(self, index):
        path = self.paths[index]
        img, _, _ = Image.open(path).convert('YCbCr').split()
        x = img.filter(ImageFilter.GaussianBlur(2))
        x = self.transform(x)
        return x / 255, self.transform(img) / 255

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    img, _, _ = Image.open('/Users/xiaohao/Documents/Python/SRCNN/BSDS300/train/15004.jpg').convert('YCbCr').split()
    plt.imshow(np.array(img) / 255)
    plt.show()
