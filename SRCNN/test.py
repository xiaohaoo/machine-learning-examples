import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset import SRCNNDataset
from srcnn import SRCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test():
    model = SRCNN().to(device)
    model.load_state_dict(torch.load("srcnn.pth"))
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(150),
        torchvision.transforms.ToTensor(),
    ])
    dataset = SRCNNDataset(transform=transform, train=False)
    data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

    for epoch in range(35):
        model.train()
        with tqdm(total=len(dataset) % 16) as t:
            t.set_description('epoch: {}/{}'.format(epoch, 35 - 1))
            for data in data_loader:
                x, labels = data
                x = x.to(device)
                labels = labels.to(device)
                output = model(x)
                batch = x.size()[0]
                for i in batch:
                    plt.figure()
                    plt.subplot(batch, 2, i * 2)
                    plt.imshow(x[i].cpu())
                    plt.subplot(batch, 2, i * 2 + 1)
                    plt.imshow(output[i].cpu())
                    plt.show()


if __name__ == '__main__':
    test()
