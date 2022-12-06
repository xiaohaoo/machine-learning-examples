import torch
import torchvision
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset import SRCNNDataset
from srcnn import SRCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(150),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = SRCNNDataset(transform=transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    for epoch in range(35):
        model.train()

        with tqdm(total=len(train_dataset) % 16) as t:
            t.set_description('epoch: {}/{}'.format(epoch, 35 - 1))
            for data in train_dataloader:
                x, labels = data
                x = x.to(device)
                labels = labels.to(device)
                output = model(x)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix(loss='{:.6f}'.format(loss.item()))
                t.update(len(x))
    torch.save(model.state_dict(), 'srcnn.pth')


if __name__ == '__main__':
    train()
